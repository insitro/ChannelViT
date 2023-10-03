import io
import time
from typing import Tuple, Union, cast

import boto3
import numpy as np


class S3SharedClient:
    def __init__(self):
        self.s3_client = boto3.client("s3")

    @staticmethod
    def get_bucket_key(
        path: str = None, bucket: str = None, key: str = None
    ) -> Tuple[str, str]:
        if path is not None:
            if bucket is not None or key is not None:
                raise ValueError("Either path is None or bucket and key are both None.")
            elif not path.startswith("s3://"):
                raise ValueError("Input path must start with s3://")

            bucket, key = path.replace("s3://", "", 1).split("/", 1)

        else:
            if bucket is None and key is None:
                raise ValueError("All inputs cannot be None.")

            bucket = cast(str, bucket)
            key = cast(str, key)

        return bucket, key

    def get_image(
        self, path: str, stop_max_attempt_number: int = 10, wait_sec: int = 3
    ):
        bucket, key = S3SharedClient.get_bucket_key(path)

        error_msg = None

        for _ in range(stop_max_attempt_number):
            try:
                s3_obj = self.s3_client.get_object(Bucket=bucket, Key=key)
                out = np.load(io.BytesIO(s3_obj["Body"].read()), allow_pickle=True)
                return out

            except Exception as e:
                error_msg = e
                time.sleep(wait_sec)
                self.s3_client = boto3.client("s3")

        print(error_msg)

        return None


s3_client = S3SharedClient()


def get_image(path):
    global s3_client
    return s3_client.get_image(path)
