import sapien
import sapien.internal_renderer as R
from sapien.utils.viewer.plugin import Plugin
import numpy as np
import time
from scipy.spatial.transform import Rotation
import torch

class TeleopDashboard(Plugin):
    """
    A SAPIEN Viewer Plugin that provides a HUD for robot teleoperation.
    """
    def __init__(self, env, manager):
        super().__init__()
        self.env = env
        self.manager = manager
        self.eef_axes = None
        self.camera_previews = []
        self.ui_window = None
        
        # Latest data from environment
        self.last_reward = 0.0
        self.is_success = False
        self.camera_width = 300.0
        self.camera_height = 225
        
        # UI overlays
        self.success_overlay = None
        self.notification_overlay = None
        
        # Track scale changes for notifications
        self.last_scale_xyz = manager.scale_xyz
        self.last_scale_rot = manager.scale_rot
        self.scale_notif_time = 0
        self.scale_notif_text = ""

    def init(self, v):
        super().init(v)
        # Add coordinate frame for End Effector (TCP)
        # We initialize it at origin; it will be moved in before_render
        self.eef_axes = self.viewer.add_coordinate_frame(sapien.Pose(), length=0.15, radius=0.01)

    def before_render(self):
        """Update 3D elements before the viewer renders the scene."""
        if self.eef_axes is None:
            return
            
        agent = self.env.unwrapped.agent
        # Support various robot implementations of TCP
        tcp_link = getattr(agent, "tcp", None)
        if tcp_link is None:
            # Common ManiSkill fallback: look for 'tcp' or 'grasp_site' or last link
            tcp_link = agent.robot.links_map.get("tcp", 
                       agent.robot.links_map.get("grasp_site", 
                       agent.robot.links[-1]))
        
        # ManiSkill 3 poses are usually batched [N, 7] or [N, 4, 4]
        pose = tcp_link.pose
        p = pose.p
        q = pose.q
        
        if isinstance(p, torch.Tensor):
            p = p[0].cpu().numpy()
            q = q[0].cpu().numpy()
        else:
            p = p[0]
            q = q[0]
            
        self.eef_axes.set_position(p)
        self.eef_axes.set_rotation(q)

    def get_ui_windows(self):
        """Create and update the 2D UI overlay windows."""
        if self.ui_window is None:
            self.ui_window = R.UIWindow().Label("Teleop Dashboard").Pos(20, 20).Size(320, 500)
            
            # Status Section
            def get_recording_text():
                if self.manager.is_saving_enabled:
                    # Flashing dot: toggle between [REC] and [   ]
                    tag = "[REC]" if int(time.time() * 2) % 2 == 0 else "[   ]"
                    return f"System Status: {tag}"
                return "System Status: [OFF]"
            
            def get_reward_text():
                return f"Current Reward: {self.last_reward:.4f}"
            
            def get_success_text():
                if self.is_success:
                    return "Task Status: [SUCCESS]"
                return "Task Status: [IN PROGRESS]"
            
            def get_lock_text():
                rot = "[ON]" if self.manager.rotation_lock else "[OFF]"
                z = "[ON]" if self.manager.z_lock else "[OFF]"
                return f"Rotation Lock: {rot} | Z-Axis Lock: {z}"

            def get_sensitivity_text():
                return f"XYZ Scale: {self.manager.scale_xyz:.3f} | Rot Scale: {self.manager.scale_rot:.3f}"

            def get_episode_info_text():
                gym_id = self.manager.gym_episode_id
                save_id = self.manager.recorder.episode_count if self.manager.recorder else "N/A"
                if isinstance(save_id, int):
                    save_id = f"{save_id:06d}"
                return f"Gym Episode: {gym_id} | Saving Episode: {save_id}"

            def get_pose_text():
                agent = self.env.unwrapped.agent
                tcp_link = getattr(agent, "tcp", agent.robot.links[-1])
                pose = tcp_link.pose
                p = pose.p[0].cpu().numpy() if isinstance(pose.p, torch.Tensor) else pose.p[0]
                q = pose.q[0].cpu().numpy() if isinstance(pose.q, torch.Tensor) else pose.q[0]
                # SciPy uses [x, y, z, w] for quat, SAPIEN uses [w, x, y, z]
                r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                euler = r.as_euler('xyz', degrees=True)
                return (f"EEF Position: {p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}\n"
                        f"EEF Rotation: {euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}")

            self.ui_window.append(
                R.UISection().Label("Status & Metrics").Expanded(True).append(
                    R.UIDisplayText().Bind(get_recording_text),
                    R.UIDisplayText().Bind(get_reward_text),
                    R.UIDisplayText().Bind(get_success_text),
                    R.UIDisplayText().Bind(get_lock_text),
                    R.UIDisplayText().Bind(get_sensitivity_text),
                    R.UIDisplayText().Bind(get_episode_info_text),
                ),
                R.UISection().Label("EEF Pose (Euler)").Expanded(True).append(
                    R.UIDisplayText().Bind(get_pose_text),
                ),
                R.UISection().Label("Camera HUD Settings").Expanded(True).append(
                    R.UISliderFloat().Label("Preview Width").Min(80).Max(400).Bind(self, "camera_width")
                ),
                R.UISection().Label("Camera HUD (Tiled)").Expanded(True)
            )
            
            # Setup Camera HUD (thumbnails) - Tiled layout
            sensors = self.env.unwrapped._sensors
            camera_items = []
            for name, cam in sensors.items():
                if hasattr(cam, "camera") and "render" not in name:
                    pic = R.UIPicture().Size(140, 105)
                    self.camera_previews.append((pic, cam))
                    camera_items.append((name, pic))
            
            # Layout cameras in rows of 2
            for i in range(0, len(camera_items), 2):
                row = R.UISameLine()
                for j in range(2):
                    if i + j < len(camera_items):
                        name, pic = camera_items[i+j]
                        container = R.UISection().Label(name).append(pic)
                        row.append(container)
                self.ui_window.append(row)

        # Update thumbnails every frame
        self.camera_height = int(self.camera_width * 0.75)
        for pic, cam in self.camera_previews:
            pic.Size(int(self.camera_width), self.camera_height)
            cam.camera.take_picture()
            # In ManiSkill 3, cam.camera is a RenderCamera wrapper. 
            # We get the first internal camera's renderer for the preview.
            sapien_cam = cam.camera._render_cameras[0]
            pic.Picture(sapien_cam._internal_renderer, "Color")

        # Handle scale change notifications
        if self.manager.scale_xyz != self.last_scale_xyz:
            self.scale_notif_text = f"XYZ Scale: {self.manager.scale_xyz:.1f}"
            self.scale_notif_time = time.time()
            self.last_scale_xyz = self.manager.scale_xyz
        if self.manager.scale_rot != self.last_scale_rot:
            self.scale_notif_text = f"Rot Scale: {self.manager.scale_rot:.1f}"
            self.scale_notif_time = time.time()
            self.last_scale_rot = self.manager.scale_rot

        return [self.ui_window] + self._get_overlays()

    def _get_overlays(self):
        overlays = []
        
        # Unified Notification Panel (Top-Center)
        if (self.is_success or self.manager.rotation_lock or self.manager.z_lock or 
            self.manager.waiting_for_skip_confirmation or (time.time() - self.scale_notif_time < 2.0)):
            if self.notification_overlay is None:
                self.notification_overlay = R.UIWindow().Label("Notifications").Pos(400, 20).Size(400, 180)
            
            self.notification_overlay.remove_children()
            
            # 1. Skip Prompt (Highest Priority during interaction)
            if self.manager.waiting_for_skip_confirmation:
                self.notification_overlay.append(R.UIDisplayText().Text("  *** SKIP EPISODE? ***"))
                self.notification_overlay.append(R.UIDisplayText().Text(" Save current data?"))
                self.notification_overlay.append(R.UIDisplayText().Text(" [Y]es / [N]o / [C]ancel"))
            else:
                # 2. Success
                if self.is_success:
                    self.notification_overlay.append(R.UIDisplayText().Text("  *** TASK SUCCESSFUL ***"))
                    self.notification_overlay.append(R.UIDisplayText().Text("   Press SPACE to Reset"))
                
                # 3. Movement Locks
                if self.manager.rotation_lock:
                    self.notification_overlay.append(R.UIDisplayText().Text(" ALERT: ROTATION LOCKED"))
                if self.manager.z_lock:
                    self.notification_overlay.append(R.UIDisplayText().Text(" ALERT: Z-AXIS TRANSLATION LOCKED"))
                
                # 4. Sensitivity Updates (Temporary)
                if time.time() - self.scale_notif_time < 2.0:
                    self.notification_overlay.append(R.UIDisplayText().Text(f" SCALE: {self.scale_notif_text}"))
            
            overlays.append(self.notification_overlay)
            
        return overlays

    def update_state(self, reward, success):
        """Update dashboard data from the main loop."""
        self.last_reward = float(reward)
        self.is_success = bool(success)
