"""
Distractor Placement Logic

Provides algorithms for placing objects evenly and randomly in the workspace.
"""

import numpy as np
from scipy.stats import qmc
from typing import Tuple, List, Optional


class UniformFrontSampler:
    """
    Samples points using Poisson Disk Sampling for "even + random" placement.
    
    Points are guaranteed to be at least `min_distance` apart and within bounds.
    """
    
    def __init__(
        self,
        x_bounds: Tuple[float, float] = (-0.2, 0.2),
        y_bounds: Tuple[float, float] = (-0.2, 0.2),
        min_distance: float = 0.08,
        seed: Optional[int] = None,
    ):
        """
        Args:
            x_bounds: (x_min, x_max) workspace bounds in X
            y_bounds: (y_min, y_max) workspace bounds in Y
            min_distance: Minimum distance between sampled points
            seed: Random seed for reproducibility
        """
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.min_distance = min_distance
        self.seed = seed
        
        # Compute normalized radius for unit square
        x_range = x_bounds[1] - x_bounds[0]
        y_range = y_bounds[1] - y_bounds[0]
        self._scale = np.array([x_range, y_range])
        self._offset = np.array([x_bounds[0], y_bounds[0]])
        
        # Radius in normalized [0,1] space
        self._normalized_radius = min_distance / max(x_range, y_range)
    
    def sample(self, n: int, max_attempts: int = 5) -> np.ndarray:
        """
        Sample n points using Poisson Disk Sampling.
        
        Args:
            n: Number of points to sample
            max_attempts: Maximum retry attempts if not enough points generated
            
        Returns:
            Array of shape (n, 2) with (x, y) positions, or fewer if impossible
        """
        rng = np.random.default_rng(self.seed)
        
        for attempt in range(max_attempts):
            try:
                # Use scipy's Poisson Disk sampler
                engine = qmc.PoissonDisk(
                    d=2, 
                    radius=self._normalized_radius,
                    seed=rng.integers(0, 2**31)
                )
                # Generate more candidates than needed
                samples = engine.random(n * 3)
                
                if len(samples) >= n:
                    # Randomly select n from available
                    indices = rng.choice(len(samples), size=n, replace=False)
                    selected = samples[indices]
                    # Map from [0,1] to workspace bounds
                    return selected * self._scale + self._offset
                    
            except Exception:
                pass
            
            # Reduce radius slightly for retry
            self._normalized_radius *= 0.9
        
        # Fallback: uniform random (not ideal but works)
        samples = rng.random((n, 2))
        return samples * self._scale + self._offset
    
    def sample_with_z(self, n: int, z_height: float = 0.05) -> np.ndarray:
        """
        Sample points and add a fixed Z height.
        
        Args:
            n: Number of points
            z_height: Height above table
            
        Returns:
            Array of shape (n, 3) with (x, y, z) positions
        """
        xy = self.sample(n)
        z = np.full((len(xy), 1), z_height)
        return np.hstack([xy, z])


class CollisionAwareSampler:
    """
    Collision-aware object placement using greedy sequential placement.
    
    Uses 2D footprints (bounding circles or AABBs) to avoid overlaps.
    Falls back to UniformFrontSampler positioning if collision-free 
    placement fails after max attempts.
    """
    
    def __init__(
        self,
        x_bounds: Tuple[float, float] = (-0.2, 0.2),
        y_bounds: Tuple[float, float] = (-0.2, 0.2),
        margin: float = 0.01,
        max_placement_attempts: int = 100,
        warn_on_collision: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            x_bounds: (x_min, x_max) workspace bounds in X
            y_bounds: (y_min, y_max) workspace bounds in Y
            margin: Extra spacing between objects (added to footprint)
            max_placement_attempts: Max random tries per object before fallback
            warn_on_collision: If True, print warning when collision-free placement fails
            seed: Random seed for reproducibility
        """
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.margin = margin
        self.max_placement_attempts = max_placement_attempts
        self.warn_on_collision = warn_on_collision
        self.rng = np.random.default_rng(seed)
        self._collision_count = 0  # Track collisions for reporting
        
    def place_objects(
        self, 
        footprints: List[dict],
        z_height: float = 0.05,
        use_z_offset: bool = False,
        z_stagger: float = 0.0,
    ) -> List[np.ndarray]:
        """
        Place objects with collision avoidance using greedy sequential placement.
        
        Args:
            footprints: List of footprint dicts, each with keys:
                - 'type': 'circle' or 'aabb'
                - 'radius': float (for circle)
                - 'half_extents': (hx, hy) tuple (for aabb)
                - 'z_offset': float (optional, distance from center to bottom)
            z_height: Base height above table for all objects
            use_z_offset: If True, use z_offset from footprint to sit objects on table
            z_stagger: If > 0, stagger Z positions to avoid initial physics collisions
            
        Returns:
            List of (x, y, z) positions for each object
        """
        placed = []  # List of (x, y, footprint_dict)
        positions = []
        self._collision_count = 0
        
        for i, footprint in enumerate(footprints):
            pos, had_collision = self._find_collision_free_position(footprint, placed)
            if had_collision:
                self._collision_count += 1
            placed.append((pos[0], pos[1], footprint))
            
            # Calculate Z position
            if use_z_offset:
                z = footprint.get('z_offset', z_height)
            else:
                z = z_height
            
            # Add stagger for drop-on-table effect
            if z_stagger > 0:
                z += i * z_stagger
            
            positions.append(np.array([pos[0], pos[1], z]))
        
        # Warn if collisions occurred
        if self.warn_on_collision and self._collision_count > 0:
            import warnings
            workspace_area = (self.x_bounds[1] - self.x_bounds[0]) * (self.y_bounds[1] - self.y_bounds[0])
            warnings.warn(
                f"CollisionAwareSampler: {self._collision_count}/{len(footprints)} objects could not be "
                f"placed collision-free. Workspace area: {workspace_area:.3f}m². Consider reducing "
                f"object count, scale, or expanding workspace bounds.",
                RuntimeWarning
            )
            
        return positions
    
    @property
    def collision_count(self) -> int:
        """Number of objects that could not be placed collision-free in last placement."""
        return self._collision_count
    
    def _find_collision_free_position(
        self, 
        footprint: dict, 
        placed: List[Tuple[float, float, dict]]
    ) -> Tuple[Tuple[float, float], bool]:
        """
        Find a collision-free position for an object.
        
        Uses random sampling with collision checks. Falls back to 
        least-colliding position if no collision-free spot found.
        
        Sampling is restricted to ensure the object stays within bounds 
        based on its bounding radius.
        
        Returns:
            Tuple of ((x, y), had_collision) where had_collision is True if 
            no collision-free spot was found.
        """
        best_pos = None
        best_collision_count = float('inf')
        
        # Restrict bounds based on object radius to ensure it stays on table
        radius = self._get_bounding_radius(footprint)
        x_min = self.x_bounds[0] + radius
        x_max = self.x_bounds[1] - radius
        y_min = self.y_bounds[0] + radius
        y_max = self.y_bounds[1] - radius
        
        # Fallback if object is too large for bounds
        if x_min >= x_max:
            x_min, x_max = (self.x_bounds[0] + self.x_bounds[1]) / 2, (self.x_bounds[0] + self.x_bounds[1]) / 2 + 0.001
        if y_min >= y_max:
            y_min, y_max = (self.y_bounds[0] + self.y_bounds[1]) / 2, (self.y_bounds[0] + self.y_bounds[1]) / 2 + 0.001

        for _ in range(self.max_placement_attempts):
            # Random candidate position within restricted bounds
            x = self.rng.uniform(x_min, x_max)
            y = self.rng.uniform(y_min, y_max)
            
            # Count collisions
            collision_count = 0
            for px, py, pf in placed:
                if self._check_collision(x, y, footprint, px, py, pf):
                    collision_count += 1
            
            if collision_count == 0:
                return (x, y), False  # Found collision-free spot
            
            if collision_count < best_collision_count:
                best_collision_count = collision_count
                best_pos = (x, y)
        
        # Return best attempt (has collisions)
        fallback = best_pos if best_pos else (
            self.rng.uniform(x_min, x_max),
            self.rng.uniform(y_min, y_max)
        )
        return fallback, True
    
    def _check_collision(
        self,
        x1: float, y1: float, f1: dict,
        x2: float, y2: float, f2: dict,
    ) -> bool:
        """
        Check if two footprints at given positions collide.
        
        Handles circle-circle, circle-aabb, and aabb-aabb cases.
        """
        t1, t2 = f1.get('type', 'circle'), f2.get('type', 'circle')
        
        if t1 == 'circle' and t2 == 'circle':
            return self._circle_circle_collision(
                x1, y1, f1.get('radius', 0.05) + self.margin,
                x2, y2, f2.get('radius', 0.05) + self.margin,
            )
        elif t1 == 'aabb' and t2 == 'aabb':
            he1 = f1.get('half_extents', (0.05, 0.05))
            he2 = f2.get('half_extents', (0.05, 0.05))
            return self._aabb_aabb_collision(
                x1, y1, he1[0] + self.margin, he1[1] + self.margin,
                x2, y2, he2[0] + self.margin, he2[1] + self.margin,
            )
        else:
            # Mixed: convert to bounding circle for simplicity
            r1 = self._get_bounding_radius(f1) + self.margin
            r2 = self._get_bounding_radius(f2) + self.margin
            return self._circle_circle_collision(x1, y1, r1, x2, y2, r2)
    
    @staticmethod
    def _circle_circle_collision(
        x1: float, y1: float, r1: float,
        x2: float, y2: float, r2: float,
    ) -> bool:
        """Check if two circles collide."""
        dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return dist_sq < (r1 + r2) ** 2
    
    @staticmethod
    def _aabb_aabb_collision(
        x1: float, y1: float, hx1: float, hy1: float,
        x2: float, y2: float, hx2: float, hy2: float,
    ) -> bool:
        """Check if two axis-aligned bounding boxes collide."""
        return (
            abs(x1 - x2) < (hx1 + hx2) and
            abs(y1 - y2) < (hy1 + hy2)
        )
    
    @staticmethod
    def _get_bounding_radius(footprint: dict) -> float:
        """Get a bounding circle radius for any footprint type."""
        if footprint.get('type') == 'circle':
            return footprint.get('radius', 0.05)
        elif footprint.get('type') == 'aabb':
            he = footprint.get('half_extents', (0.05, 0.05))
            return np.sqrt(he[0]**2 + he[1]**2)  # Diagonal half-length
        else:
            return 0.05  # Default fallback
