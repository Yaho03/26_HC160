# Security, Ethics, and Limitations

## 1. Research-use boundary

This project is limited to authorized defensive security research. Attack code is evaluated only against project-controlled models, data, and demonstration environments. It must not be used to target third-party authentication systems.

## 2. Biometric data

Face images and embeddings are sensitive biometric artifacts. The project follows:

- data minimization;
- pseudonymous identities in committed metadata;
- encrypted external storage where available;
- role-limited access;
- explicit retention and deletion dates;
- no raw face or embedding commit to Git;
- no public release of checkpoints or generated images without review.

## 3. Dataset and model licenses

Every dataset and pretrained weight records source and license terms. Code licensing does not grant permission to redistribute face images or third-party weights. Release review treats these as separate artifacts.

## 4. Claims and limitations

- LFW experiments do not validate real financial customers or environments.
- Public pretrained models may have identity overlap with public benchmarks.
- Current datasets do not support population fairness claims.
- A Python webcam prototype cannot attest the OS, driver, camera, or virtual-camera source.
- Simulation-based temporal results do not establish replay resistance.
- Attack and defense effectiveness changes with model, preprocessing, serialization, and threshold.

## 5. Fairness

Reports state when demographic attributes are unavailable or insufficient. No subgroup is described as secure or insecure without adequate sample size, consent, and documented methodology.

## 6. Generative AI

The approved default scope is defensive purification. Generative impersonation, identity transfer, or deepfake construction is out of scope. Purification must measure identity drift and clean false rejection, not only attack removal.

## 7. Release checklist

- no raw face images, embeddings, local paths, credentials, or private URLs;
- dataset and model license fields complete;
- result rows resolve to immutable run IDs;
- limitations and non-production disclaimer present;
- attack artifacts and code have an authorized research-use context;
- no regenerated result silently replaces a historical artifact.
