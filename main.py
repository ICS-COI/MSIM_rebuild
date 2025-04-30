import array_illumination as ai

if __name__ == '__main__':

    ai.get_lattice_vectors(
        calibration="data/lake.tif",
        # background="data/lake_background.tif",
        extent=5,  # 寻找傅里叶尖峰时一个点的覆盖范围
        num_spikes=300,  # 寻找傅里叶尖峰时的峰值数量
        tolerance=3.5,  # 傅里叶基向量所得晶格点与尖峰对应的容差
        num_harmonics=3,  # 傅里叶基向量的最小阶数
        show_ratio=1,
        scan_dimensions=(16, 14),
        verbose=True,
        display=True,
        animate=False,  # 动画显示傅里叶空间的峰值寻找过程
        show_interpolation=False,  # 滤波的中间过程
        show_lattice=True,
        record_parameters=True
    )

    ai.get_lattice_vectors(
        calibration="data/36-5-224frames-rotate.tif",
        # background="data/224_1000_1000_background.tif",
        extent=4,  # 寻找傅里叶尖峰时一个点的覆盖范围
        num_spikes=300,  # 寻找傅里叶尖峰时的峰值数量
        tolerance=3.,  # 傅里叶基向量所得晶格点与尖峰对应的容差
        num_harmonics=3,  # 傅里叶基向量的最小阶数
        show_ratio=0.1, # 傅里叶图展示大小
        low_pass_filter =0.5,
        scan_dimensions=(16, 14),
        dot_size_show = 3,
        verbose=True,
        display=True,
        animate=False,  # 动画显示傅里叶空间的峰值寻找过程
        show_interpolation=False,  # 滤波的中间过程
        show_lattice=True,
        record_parameters=True
    )

    # ai.get_lattice_vectors(
    #     calibration_name="data/24-2-5frames.tif",
    #     result_path=result_path,
    #     extent=8,  # 寻找傅里叶尖峰时一个点的覆盖范围
    #     num_spikes=60,  # 寻找傅里叶尖峰时的峰值数量
    #     tolerance=3.,  # 傅里叶基向量所得晶格点与尖峰对应的容差
    #     num_harmonics=3,  # 傅里叶基向量的最小阶数
    #     show_ratio=0.25,
    #     low_pass_filter =0.5,
    #     verbose=True,
    #     display=True,
    #     animate=False,  # 动画显示傅里叶空间的峰值寻找过程
    #     show_interpolation=False,
    # )