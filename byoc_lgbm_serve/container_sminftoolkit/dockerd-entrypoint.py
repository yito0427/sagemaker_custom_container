from sagemaker_inference import model_server
print('========== This is START of dockerd-entrypoint.py =============')
model_server.start_model_server()
print('========== This is END of dockerd-entrypoint.py =============')