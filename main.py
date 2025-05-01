import os
import pprint
from datetime import datetime
import array_illumination as ai

if __name__ == '__main__':
    calibration_filename = "data/lake.tif"
    background_filename = "data/lake_background.tif"
    data_filenames_list = ["data/lake.tif",
                           ]
    scan_dimensions = (16, 14)
    steps = scan_dimensions[0] * scan_dimensions[1]

    filename = os.path.splitext(os.path.basename(calibration_filename))[0]
    # timestamp = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
    timestamp = datetime.now().strftime("_%Y%m%d")
    result_path = os.path.join(os.path.join(os.getcwd(), "result"), filename + timestamp)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f"result_path: {result_path}\n")

    img_shape, lattice_vectors, shift_vector, offset_vector, intensities_position, background_frame = ai.get_lattice_vectors(
        calibration=calibration_filename,
        background=background_filename,
        result_path=result_path,
        extent=5,  # 寻找傅里叶尖峰时一个点的覆盖范围
        num_spikes=300,  # 寻找傅里叶尖峰时的峰值数量
        tolerance=3.5,  # 傅里叶基向量所得晶格点与尖峰对应的容差
        num_harmonics=3,  # 傅里叶基向量的最小阶数
        show_ratio=1,
        scan_dimensions=scan_dimensions,
        verbose=False,
        display=False,
        animate=False,  # 动画显示傅里叶空间的峰值寻找过程
        show_interpolation=False,  # 滤波的中间过程
        show_lattice=False,
        record_parameters=True
    )

    # img_shape, lattice_vectors, shift_vector, offset_vector, intensities_position, background_frame =ai.get_lattice_vectors(
    #     calibration="data/36-5-224frames-rotate.tif",
    #     background="data/224_1000_1000_background.tif",
    #     extent=4,  # 寻找傅里叶尖峰时一个点的覆盖范围
    #     num_spikes=300,  # 寻找傅里叶尖峰时的峰值数量
    #     tolerance=3.,  # 傅里叶基向量所得晶格点与尖峰对应的容差
    #     num_harmonics=3,  # 傅里叶基向量的最小阶数
    #     show_ratio=0.1, # 傅里叶图展示大小
    #     low_pass_filter =0.5,
    #     scan_dimensions=(16, 14),
    #     dot_size_show = 5,
    #     verbose=False,
    #     display=False,
    #     animate=False,  # 动画显示傅里叶空间的峰值寻找过程
    #     show_interpolation=False,  # 滤波的中间过程
    #     show_lattice=False,
    #     record_parameters=True
    # )

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

    print("\nLattice vectors:")
    for v in lattice_vectors:
        print(v)
    print("\nShift vector:")
    pprint.pprint(shift_vector)
    print("\nInitial position:")
    print(offset_vector)

    # 定义新的笛卡尔网格
    zPix, xPix, yPix, = img_shape
    new_grid_xrange = 0, xPix - 1, 2 * xPix
    new_grid_yrange = 0, yPix - 1, 2 * yPix

    num_processes = 1
    for f in data_filenames_list:
        # print(f)

        def profile_me():
            ai.reconstruct_image_parallel(
                data_filename=f,
                calibration_filename=calibration_filename,
                background_filename=background_filename,
                xPix=xPix, yPix=yPix, zPix=zPix, steps=steps,
                lattice_vectors=lattice_vectors,
                offset_vector=offset_vector,
                shift_vector=shift_vector,
                new_grid_xrange=new_grid_xrange, new_grid_yrange=new_grid_yrange,
                result_path=result_path,
                num_processes=num_processes,
                window_footprint=10,
                aperture_size=3,
                make_widefield_image=False,
                # make_confocal_image=False,  # Broken, for now
                verbose=True,
                show_steps=False,  # For debugging
                show_slices=False,  # For debugging
                intermediate_data=False,  # Memory hog, leave 'False'
                normalize=False,  # Of uncertain merit, leave 'False' probably
                display=False,
                cover=True, # 如果已经存在重建图片，是否重新计算
            )

        if num_processes == 1:
            import cProfile
            import pstats

            cProfile.run('profile_me()', 'profile_results')

            try:
                p = pstats.Stats('profile_results')
                p.strip_dirs().sort_stats(-1).print_stats()
                p.sort_stats('cumulative').print_stats(20)
            except ImportError:
                pass
        else:
            profile_me()

    # ai.join_enderlein_images(
    #     data_filenames_list,
    #     new_grid_xrange, new_grid_yrange,
    #     join_widefield_images=False)
