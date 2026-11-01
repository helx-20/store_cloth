# store_cloth — accepted two-garment anchor (2026-11-02)

User-approved visual baseline: video `snapshot/view920/diagnostic920.mp4` (6 s, 60 fps, H.264/avc1, yuv420p, faststart). The user accepted this version and requested expansion to more garments after creating this GitHub anchor.

## Exact baseline

- Connected single-sheet torso and sleeves; complete offline folding, five states in `snapshot/candidate758_two_local_collapses/prefold_mesh.npz`.
- First garment: run839, 2 s. Second garment: run841, 2 s, continued by run844 and run849 to 6 s after its release. Release gap approximately 5 cm above the clipped local stack surface.
- Lower garments retain actual position, velocity and bounded bend memory. Nonzero membrane stiffness, self-contact, friction, and `include_bending=True`.
- 64 iterations, four substeps/frame at 60 fps, friction epsilon 0.001. Exact q/v joins at second-item times 2 and 4 s; collision/solver history was rebuilt at those joins.
- Single-sheet geometry, authored crease targets, and bounded evolving rest-curvature memory are explicit uncalibrated approximations.

## Contents and provenance

`snapshot/` preserves original relative paths for the runtime scripts, config, asset, input, recorded trajectories, checkpoints, renderer and manifests. `snapshot/solver_bend_scaled/newton` contains the isolated Newton source actually used, rather than assuming an upstream release is identical. `environment_versions.json` records installed distribution versions; it is evidence, not a portable lockfile. `SHA256SUMS.json` binds every packaged artifact (excluding itself and Git metadata).

No credentials, virtual environments, unrelated projects, failed exploratory runs, or complete shared progress log are included.

## Inspect and reproduce

First verify the SHA256 manifest. Recorded trajectories can be inspected without running simulation. For rendering, install the recorded NumPy, Pillow and imageio-ffmpeg versions in an isolated Python environment and run `python snapshot/render920.py` from a fresh working copy in which `snapshot/view920` has first been moved aside; the renderer deliberately refuses to overwrite existing output.

Physics scripts preserve original experiment names and refuse to overwrite run directories. Reproduction requires a separate scratch copy, the recorded Python dependencies, CUDA-capable hardware and this vendored Newton tree. `kernels589.py` currently selects physical GPU index 1: review allocation before running. Do not launch these scripts on another user's allocated GPU. No automatic physics launch occurs on checkout. Original run outputs are evidence and should be preserved.

## Acceptance scope and known limitations

The user accepted the appearance and authorized increasing garment count. This is not a certification of continuous nonintersection, material calibration or equilibrium. Broad transient wrinkles relax; table reweighting, increased flat-panel bending stiffness and local refinement did not yield a demonstrated visible improvement and are not part of this anchor. Finite audits found zero transverse intersections at checked states, but small layer gaps and localized high strain remain. At second-item 6 s, upper/lower RMS speeds are approximately 0.975/0.544 mm/s; local maximum stretches approximately 4.466/4.416. Sampled checks do not prove whole-surface clearance or continuous collision freedom. No frictional-slip certification is claimed.

Next development priority: add the third garment from the accepted run849 checkpoint while keeping both lower garments dynamic. Do not restart the discarded wrinkle parameter sweep or block expansion on broad material calibration.
