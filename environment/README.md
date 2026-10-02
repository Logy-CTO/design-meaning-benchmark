# Environments

The root requirements.txt supports analysis from saved predictions on a CPU.
original_recorded_requirements.txt preserves the available version specification from the study archive; it is not a complete machine lockfile.
Generation, model inference and training require separate model access, API credentials where applicable, and compatible CUDA/PyTorch hardware.
The nested training runs used mistral-common==1.10.0. Install dependencies from their official distributions. Base-model weights and server caches are not included.
