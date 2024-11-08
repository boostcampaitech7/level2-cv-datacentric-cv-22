import os

def make_directory(directory_path):
    """
        디렉토리 경로가 존재하지 않으면 해당 경로에 디렉토리를 생성하는 함수.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")


def is_file_exist(file_path):
    """
        파일 존재 유무 확인
    """
    return os.path.exists(file_path)