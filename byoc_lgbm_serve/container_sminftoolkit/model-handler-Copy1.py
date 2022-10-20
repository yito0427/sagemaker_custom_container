from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors

class DefaultPytorchInferenceHandler(default_inference_handler.DefaultInferenceHandler):

    def default_model_fn(self, model_dir, context=None):
        """Loads a model. For PyTorch, a default function to load a model cannot be provided.
        Users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.
            context (obj): the request context (default: None).

        Returns: A PyTorch model.
        """
        raise NotImplementedError(textwrap.dedent("""
        Please provide a model_fn implementation.
        See documentation for model_fn at https://github.com/aws/sagemaker-python-sdk
        """))

    def default_input_fn(self, input_data, content_type, context=None):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type
            context (obj): the request context (default: None).

        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
        """
        return decoder.decode(input_data, content_type)

    def default_predict_fn(self, data, model, context=None):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn
            context (obj): the request context (default: None).

        Returns: a prediction
        """
        return model(input_data)

    def default_output_fn(self, prediction, accept, context=None):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized
            context (obj): the request context (default: None).

        Returns: output data serialized
        """
        return encoder.encode(prediction, accept)