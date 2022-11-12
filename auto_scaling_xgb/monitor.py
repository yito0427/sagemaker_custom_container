import boto3
import time


sm_client = boto3.client(service_name='sagemaker')

endpoint_name = 'xgboost-reg2022-11-11-02-36-05' # エンドポイント名を入力

response = sm_client.describe_endpoint(EndpointName=endpoint_name)
status = response['EndpointStatus']
print("Status: " + status)

while (status=='Updating')or(status=='InService'):
    #print("Status: " + status)
    time.sleep(1)
    response = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    instance_count = response['ProductionVariants'][0]['CurrentInstanceCount']
    #print(f"Status: {status}")
    #print(f"Current Instance count: {instance_count}")
    print(f"Status: {status}, CurrentInstance count: {instance_count}")
