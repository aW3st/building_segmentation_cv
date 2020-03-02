# Imports the Google Cloud client library

from google import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name=None):
    """
    Uploads a file to the bucket.
    
    Upload to source file name by default if destination is not specified.
    """
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    print(source_file_name)

    # What if no destination name is given?
    if destination_blob_name is None:
        destination_blob_name = source_file_name
        print(destination_blob_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


if __name__ == "__main__":

    # print('Testing Google Cloud Upload')
    # upload_blob('building-segmentation-cv', source_file_name='data/doge.jpg')

    print("Downloading images from Gcloud")
    download_blob(bucket_name=, source_blob_name='DD-building-segmentation', destination_file_name='/tmp/images')