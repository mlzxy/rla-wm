import sapien
import sapien.internal_renderer as R
from sapien.utils.viewer.plugin import Plugin
import numpy as np

class CameraConfigPlugin(Plugin):
    """
    A SAPIEN Viewer Plugin for configuring camera parameters (FOV, Pose).
    """
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.ui_window = None
        
        # State
        self.selected_camera_name = ""
        self.camera_sensors = {}
        self.step_size = 0.05
        self.angle_step = 0.05
        
        # Current camera sensor object
        self.current_sensor = None
        self.fov_value = 1.0

    def init(self, v):
        super().init(v)
        # Scan for cameras
        self._refresh_cameras()

    def _refresh_cameras(self):
        """Scan env sensors for cameras."""
        self.camera_sensors = {}
        if hasattr(self.env.unwrapped, "_sensors"):
            for name, sensor in self.env.unwrapped._sensors.items():
                if "camera" in name.lower() and hasattr(sensor, "camera"):
                    self.camera_sensors[name] = sensor
        
        # Select first available if none selected
        if self.camera_sensors and not self.selected_camera_name:
            self.selected_camera_name = list(self.camera_sensors.keys())[0]
            self.current_sensor = self.camera_sensors[self.selected_camera_name]
            self.fov_value = self.current_sensor.camera.fovx

    def _update_selection(self, name):
        if name in self.camera_sensors:
            self.selected_camera_name = name
            self.current_sensor = self.camera_sensors[name]
            self.fov_value = self.current_sensor.camera.fovx

    def _apply_fov(self):
        if self.current_sensor:
            # Clamp FOV
            val = max(0.1, min(np.pi - 0.1, self.fov_value))
            self.current_sensor.camera.set_fovx(val, compute_y=True)

    def _move_camera(self, x, y, z):
        """Move camera in its own local frame."""
        if self.current_sensor:
            # SAPIEN Camera convention: Forward is +X, Left is +Y, Up is +Z.
            pose = self.current_sensor.camera.local_pose
            delta_pose = sapien.Pose(p=[x, y, z], q=[1, 0, 0, 0])
            new_pose = pose * delta_pose
            self.current_sensor.camera.local_pose = new_pose

    def _rotate_camera(self, pitch, yaw, roll):
        """Rotate camera in its own frame."""
        # pitch (around X), yaw (around Y), roll (around Z) in SAPIEN camera frame?
        # SAPIEN Camera Frame: X=Right, Y=Up, Z=Back.
        if self.current_sensor:
            pose = self.current_sensor.camera.local_pose
            from scipy.spatial.transform import Rotation
            # Euler angles
            delta_rot = Rotation.from_euler("xyz", [pitch, yaw, roll]).as_quat()
            # SciPy: x,y,z,w -> SAPIEN: w,x,y,z
            delta_q = [delta_rot[3], delta_rot[0], delta_rot[1], delta_rot[2]]
            
            delta_pose = sapien.Pose(p=[0,0,0], q=delta_q)
            new_pose = pose * delta_pose
            self.current_sensor.camera.local_pose = new_pose

    def _print_config(self):
        if self.current_sensor:
            pose = self.current_sensor.camera.local_pose
            # ManiSkill poses are batched (B, ...). Take the first env for printing.
            p = pose.p
            mat = pose.to_transformation_matrix()
            
            eye = p
            # SAPIEN Camera convention: Forward is +X. 
            # We calculate a target 1 unit away along the forward axis.
            forward = mat[:3, 0]
            target = eye + forward
            
            # Convert to numpy for cleaner printing if they are tensors
            if hasattr(eye, "detach"): eye = eye.detach().cpu().numpy()
            if hasattr(target, "detach"): target = target.detach().cpu().numpy()
            
            print(f"\n--- Camera Config: {self.selected_camera_name} ---")
            print(f"eye: [{eye[0]:.4f}, {eye[1]:.4f}, {eye[2]:.4f}]")
            print(f"target: [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
            print(f"fov: {self.fov_value:.4f}")
            print("-------------------------------------------\n")

    def get_ui_windows(self):
        if self.ui_window is None:
            self.ui_window = R.UIWindow().Label("Camera Config").Pos(20, 300).Size(300, 600)
            self._camera_preview = R.UIPicture().Size(280, 280)
            self._camera_list_section = R.UISection().Label("Select Camera").Expanded(True)

        # Clear and Rebuild List (stateless UI style for the list part)
        # Note: In SAPIEN UI, we usually append once and update properties. 
        # But for valid selection state visualization, we might need to clear children if supported
        # or just update them. SAPIEN UI wrappers might not support clear easily?
        # Actually `dashboard.py` uses `remove_children()` on UIWindow/UISection.
        
        # 1. Camera List Section
        self._camera_list_section.remove_children()
        names = list(self.camera_sensors.keys())
        names.sort()
        
        for name in names:
            is_selected = (name == self.selected_camera_name)
            # Emulate selection with label formatting or check if UISelectable supports state
            label = f"[{'x' if is_selected else ' '}] {name}"
            # Using UIButton for robustness if UISelectable API is unknown
            # But user asked for dropdown/list.
            # Let's try UISelectable if we can, or just Button with visual cue
            
            # Using a closure to capture name
            def select_cb(n=name):
                # The callback might receive arguments (the widget), allow *args
                return lambda *args: self._update_selection(n)

            self._camera_list_section.append(
                R.UIButton().Label(label).Callback(select_cb())
            )

        # Rebuild Window
        self.ui_window.remove_children()
        
        self.ui_window.append(
            self._camera_list_section,
            R.UISection().Label("Preview").Expanded(True).append(self._camera_preview),
            R.UISection().Label("FOV").Expanded(True).append(
                 R.UISliderFloat().Label("FOV (rad)").Min(0.1).Max(3.0).Bind(self, "fov_value").Callback(lambda *args: self._apply_fov())
            ),
            R.UISection().Label("Move (Local Frame)").Expanded(True).append(
                R.UISliderFloat().Label("Step Size").Min(0.01).Max(0.5).Bind(self, "step_size"),
                R.UIDisplayText().Text("Position (WASDQE style):"),
                R.UISameLine().append(
                    R.UIButton().Label("Forward (+X)").Callback(lambda *a: self._move_camera(self.step_size, 0, 0)),
                    R.UIButton().Label("Back (-X)").Callback(lambda *a: self._move_camera(-self.step_size, 0, 0))
                ),
                R.UISameLine().append(
                    R.UIButton().Label("Left (+Y)").Callback(lambda *a: self._move_camera(0, self.step_size, 0)),
                    R.UIButton().Label("Right (-Y)").Callback(lambda *a: self._move_camera(0, -self.step_size, 0))
                ),
                R.UISameLine().append(
                    R.UIButton().Label("Up (+Z)").Callback(lambda *a: self._move_camera(0, 0, self.step_size)),
                    R.UIButton().Label("Down (-Z)").Callback(lambda *a: self._move_camera(0, 0, -self.step_size))
                )
            ),
            R.UISection().Label("Rotate (Local Frame)").Expanded(True).append(
                R.UISliderFloat().Label("Ang Step").Min(0.01).Max(0.5).Bind(self, "angle_step"),
                 R.UISameLine().append(
                     R.UIButton().Label("Pitch Up").Callback(lambda *a: self._rotate_camera(self.angle_step, 0, 0)),
                     R.UIButton().Label("Pitch Down").Callback(lambda *a: self._rotate_camera(-self.angle_step, 0, 0))
                 ),
                 R.UISameLine().append(
                     R.UIButton().Label("Yaw Left").Callback(lambda *a: self._rotate_camera(0, self.angle_step, 0)),
                     R.UIButton().Label("Yaw Right").Callback(lambda *a: self._rotate_camera(0, -self.angle_step, 0))
                 )
            ),
            R.UISection().Label("Output").Expanded(True).append(
                R.UIButton().Label("Print Config to Console").Callback(lambda *a: self._print_config())
            )
        )

        # Update Preview
        if self.current_sensor and hasattr(self.current_sensor, "camera"):
            cam = self.current_sensor.camera
            cam.take_picture()
            # access internal renderer
            if hasattr(cam, "_render_cameras") and len(cam._render_cameras) > 0:
                sapien_cam = cam._render_cameras[0]
                self._camera_preview.Picture(sapien_cam._internal_renderer, "Color")

        return [self.ui_window]
