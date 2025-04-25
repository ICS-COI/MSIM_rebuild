# import cv2
# import numpy as np
# import array_detect as ar
import os
from datetime import datetime
import array_illumination as ai

if __name__ == '__main__':
    image_path = "data/lake.tif"
    # image_path = "data/24-2-5frames.tif"

    filename = os.path.splitext(os.path.basename(image_path))[0]
    # timestamp = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
    timestamp = datetime.now().strftime("_%Y%m%d")
    result_path = os.path.join(os.path.join(os.getcwd(), "result"), filename + timestamp)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f"result_path: {result_path}")

    ai.get_lattice_vectors(
        calibration_name=image_path,
        result_path=result_path,
        extent=5,  # 寻找傅里叶尖峰时一个点的覆盖范围
        num_spikes=60,  # 寻找傅里叶尖峰时的峰值数量
        tolerance=3.,  # 傅里叶基向量所得晶格点与尖峰对应的容差
        num_harmonics=3,  # 傅里叶基向量的最小阶数
        show_ratio=1,
        verbose=True,
        display=True,
        animate=False,  # 动画显示傅里叶空间的峰值寻找过程
        show_interpolation=False,
    )

    # ai.get_lattice_vectors(
    #     calibration_name=image_path,
    #     result_path=result_path,
    #     extent=8,  # 寻找傅里叶尖峰时一个点的覆盖范围
    #     num_spikes=60,  # 寻找傅里叶尖峰时的峰值数量
    #     tolerance=3.,  # 傅里叶基向量所得晶格点与尖峰对应的容差
    #     num_harmonics=3,  # 傅里叶基向量的最小阶数
    #     show_ratio=0.25,
    #     verbose=True,
    #     display=True,
    #     animate=False,  # 动画显示傅里叶空间的峰值寻找过程
    #     show_interpolation=False,
    # )


    # _, image_all = cv2.imreadmulti(image_path, flags=cv2.IMREAD_UNCHANGED)
    # image = np.array(image_all)
    # ar.detect_dot_centers(image, weighted=True, verbose=False, show=True)
