FROM public.ecr.aws/lambda/python:3.13

RUN pip install "numpy<2" "onnxruntime==1.16.3" pillow

COPY lambda_function.py .
COPY hair_classifier_v1.onnx .
COPY hair_classifier_v1.onnx.data .

CMD [ "lambda_function.lambda_handler" ]