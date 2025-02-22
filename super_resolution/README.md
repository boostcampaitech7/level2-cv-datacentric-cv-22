# Super Resolution

### 사용법
* **Baseline code** : [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch/tree/master)

    ```
    git clone https://github.com/sanghyun-son/EDSR-PyTorch.git
    ```

* super resolution을 진행할 이미지를 준비한다.

    * divide_image.py를 사용하여 이미지를 분할한다. (GPU 메모리 이슈로 인해 이미지를 분할해서 super resolution을 진행함)

    * 이 때 분할한 이미지는 test 폴더에 넣는다. 

        ```
        python tools/divide_image.py --input_dir 원본 데이터 경로 --output_dir test 폴더 경로 --split_type 16
        ```

* Baseline code에서 제공하는 모델을 사용해서 super resolution을 진행한다.

* super resolution이 적용된 분할된 이미지를 하나의 이미지로 합친다.

    * image_concat.py를 사용한다.

        ```
        python tools/image_concat.py --input_dir test 폴더 경로 --output_dir 합친 이미지 경로
        ```

* super resolution이 적용된 이미지에 맞게 annotation 좌표들을 수정한다.

    * annotation_x2.py를 사용한다.
