import boto3
import json
class SNSPublish:
    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, region_name: str, aws_sns_arn: str):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.aws_sns_arn = aws_sns_arn
        self.sns_client = boto3.client("sns", region_name=self.region_name)

    def publish(self, message: dict):
        # we convert message to json , because sns only accepts json messages and out input from api is a dictionary.
        self.sns_client.publish(TopicArn=self.aws_sns_arn, Message=json.dumps(message))