import io
import time
from typing import Tuple, Union, cast

import boto3
import numpy as np
from botocore.config import Config
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def _get_bucket_key(
    path: Union[str, None] = None,
    bucket: Union[str, None] = None,
    key: Union[str, None] = None,
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


class S3Dataset(Dataset):
    def __init__(self):
        self.s3_client = None

    def get_image(self, path, max_attempts=5, wait_sec=2):
        error_code = None
        # for _ in range(max_attempts):
        attempt = 0
        while True:
            try:
                if self.s3_client is None:
                    self.s3_client = boto3.client(
                        "s3", config=Config(retries=dict(max_attempts=10))
                    )

                bucket, key = _get_bucket_key((path))
                s3_obj = self.s3_client.get_object(Bucket=bucket, Key=key)
                if key.endswith(".npy"):
                    # read numpy inputs
                    out = np.load(io.BytesIO(s3_obj["Body"].read()), allow_pickle=True)
                elif key.endswith(".png") or key.endswith(".jpg"):
                    # read png images
                    out = Image.open(io.BytesIO(s3_obj["Body"].read())).convert("RGB")
                else:
                    raise ValueError(f"Unsupported file format {key}")

                return out

            except Exception as e:
                error_code = e
                time.sleep(wait_sec)
                self.s3_client = None
                if attempt > 1:
                    print(f"Attempt {attempt} ", e)
                attempt += 1

        print(error_code)
        return None

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)
