from sagemaker_inference import model_server
print('========== This is dockerd-entrypoint.py =============')
model_server.start_model_server()