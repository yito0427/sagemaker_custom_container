from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer
#from sagemaker_pytorch_serving_container.default_inference_handler import DefaultPytorchInferenceHandler
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler

class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md
    """
    def __init__(self):
        #transformer = Transformer(default_inference_handler=DefaultPytorchInferenceHandler())
        transformer = Transformer(default_inference_handler=DefaultInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)