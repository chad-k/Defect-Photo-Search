# Defect Photo Search — Streamlit Cloud edition

This is a separate hosted edition. Includes the same 100 photorealistic synthetic reference images from the updated local demo. CLIP is selected by default. Photos are processed on the hosting server; this is not an offline application.

## Update your existing GitHub deployment

Extract the ZIP on your computer. Upload the CONTENTS of Defect_Photo_Search_Cloud to the folder in your repository containing app.py. Do not upload the ZIP itself.

Replace app.py, engine.py, demo_data.py, requirements.txt and .streamlit/config.toml. Keep the complete demo_realistic folder; if you already uploaded that 100-image folder, there is no need to upload it again. The small Cloud_Update ZIP includes just code/config/instructions for that situation. Do not copy these cloud files into your offline installation.

Remove any old environment.yml, Pipfile, pyproject.toml or uv.lock used as a competing dependency configuration for this app. Do not upload models/, local_cases/, private photos, API keys, or a Python environment. The cloud app does not need requirements-ai.txt, download_model.py or run_dashboard.py. Those old files can be removed from the deployment repository.

In Streamlit Community Cloud, choose your repository and branch, set the entrypoint to app.py (or its full relative path if nested), and choose Python 3.11. Commit the updated requirements.txt and let Streamlit rebuild. Reboot through Manage app if necessary. If the existing app uses a different Python version and you cannot change it there, redeploy it with Python 3.11.

The cloud configuration omits the local launcher's loopback binding. Launch locally for testing with `python -m streamlit run app.py` after installing the cloud requirements in a separate environment.

## CLIP setup and first search

No terminal commands or API key are required on the hosting service. requirements.txt installs CPU-only PyTorch and the CLIP libraries. First search downloads the pinned official openai/clip-vit-base-patch32 model (roughly 605 MB for its checkpoint) from Hugging Face, loads its vision encoder, then builds reference features. Subsequent searches reuse the model and public reference features in memory. A new server instance may need to repeat setup. Network restrictions or hosting memory limits can still prevent loading; inspect Streamlit logs if it fails.

The pinned model release has a PyTorch checkpoint, not a safetensors file. It is loaded from the official repository at a fixed commit, with weights_only=True and no remote code execution. The app does not upload images to Hugging Face or use a hosted inference API. URLs are only used to obtain model assets.

## Temporary reference cases

Add a case stores a reference in the current Streamlit session, limited to 20 images / 50 MB. Choose My session cases to search them. Each session gets its own list; uploaded images do not enter the shared reference-feature cache or get written into a shared case folder. Clear session cases removes the reference list and current results. Query uploader widgets can still hold a query until replaced or the session ends. Refresh, disconnect or server restart may lose temporary cases. This is not durable storage or a production multi-user records system. Hosting-provider retention policies also apply.

## Demonstration limits

All included images and case histories are synthetic. No exact duplicates of the separate 100-image test pack are included. A reference sample excludes itself from search results. Similarity scores are not probabilities, diagnoses, or verified causes. Background and part geometry can dominate CLIP rankings. Real-world retrieval accuracy and concurrent-user capacity have not been validated.

Documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
CLIP: https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip
Model: https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
