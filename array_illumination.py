import os, sys, pickle, pprint, subprocess, time, random
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, interpolation
from scipy.signal.windows import hann, gaussian
import cv2
from datetime import datetime
import utils


def get_lattice_vectors(
        calibration=None,
        background=None,
        result_path=None,
        extent=8,  # 寻找傅里叶尖峰时一个点的覆盖范围，需调整 8
        num_spikes=60,  # 寻找傅里叶尖峰时的峰值数量，需调整
        tolerance=3.,  # 傅里叶基向量所得晶格点与尖峰对应的容差
        num_harmonics=3,  # 傅里叶基向量的最小阶数
        show_ratio=0.25,  # 显示傅里叶空间的峰值的图像比例，为了更好地看清低频点 0.25
        low_pass_filter=0.1,  # 低通滤波的截止频率（高频有错位峰值）
        outlier_phase=1.,
        calibration_window_size=10,
        scan_type='dmd',
        scan_dimensions=None,
        dot_size_show=1,  # 显示晶格图案的点的大小
        verbose=True,
        display=True,
        animate=False,  # 动画显示傅里叶空间的峰值寻找过程
        show_interpolation=False,
        show_calibration_steps=False,
        show_lattice=False,
        record_parameters=False):
    """
    由校准图像计算出照明晶格参数（给定一个扫描场图像栈，找出照明晶格图案的基向量。）

    :param calibration: 校准图像的路径
    :param background: 背景图像的路径
    :param result_path: 结果保存的路径
    :param extent: 寻找傅里叶尖峰时一个点的覆盖范围，需调整
    :param num_spikes: 寻找傅里叶尖峰时的峰值数量，需调整
    :param tolerance: 傅里叶基向量所得晶格点与尖峰对应的容差
    :param num_harmonics: 傅里叶基向量的最小阶数
    :param show_ratio: 显示傅里叶空间的峰值的图像比例，为了更好地看清低频点
    :param low_pass_filter: 低通滤波的截止频率（高频有错位峰值）
    :param outlier_phase: 傅里叶空间的峰值的相位偏移，用于去除异常值
    :param calibration_window_size: 校准图像的窗口大小
    :param scan_type: 扫描类型，'dmd'
    :param scan_dimensions: 扫描图像的尺寸
    :param dot_size_show: 显示晶格图案的点的大小
    :param verbose: 是否打印详细信息
    :param display: 是否展示图像
    :param animate: 是否显示傅里叶峰值查找的动画
    :param show_interpolation: 是否显示寻找精确最大值的插值过程
    :param show_calibration_steps: 是否显示校准过程
    :param show_lattice: 是否展示参数确定后的晶格图案
    :param record_parameters: 是否记录参数
    :return:
    """
    if result_path is None:
        filename = os.path.splitext(os.path.basename(calibration))[0]
        # timestamp = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
        timestamp = datetime.now().strftime("_%Y%m%d")
        result_path = os.path.join(os.path.join(os.getcwd(), "result"), filename + timestamp)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if verbose:
            print(f"result_path: {result_path}\n")

    calibration_all = load_image_data(calibration)
    zPix, xPix, yPix = calibration_all.shape

    print("Detecting calibration illumination lattice parameters...")

    # 粗略估计晶格向量
    fft_data_folder, fft_abs, fft_avg = get_fft_abs(calibration, calibration_all, result_path,
                                                    verbose=verbose)  # DC term at center
    filtered_fft_abs = spike_filter(fft_abs, display=False)

    # 在傅里叶域中寻找候选尖峰
    if verbose:
        print("Finding Fourier-space spikes...")
    coords = find_spikes(fft_abs, filtered_fft_abs, extent=extent, num_spikes=num_spikes,
                         low_pass_filter=low_pass_filter,
                         show_ratio=show_ratio, display=display,
                         animate=animate)

    # 用这些候选尖峰来确定傅里叶空间晶格
    if verbose:
        print("Finding Fourier-space lattice vectors...")
    basis_vectors = get_basis_vectors(fft_abs, coords, tolerance=tolerance, num_harmonics=num_harmonics,
                                      verbose=verbose)
    if verbose:
        print("Fourier-space lattice vectors:")
        for v in basis_vectors:
            print(v, "(Magnitude", np.sqrt((v ** 2).sum()), ")")

    # 通过约束傅里叶空间向量的和为零来修正这些向量。
    error_vector = sum(basis_vectors)
    corrected_basis_vectors = [v - ((1. / 3.) * error_vector) for v in basis_vectors]
    if verbose:
        print("Fourier-space lattice vector triangle sum:", error_vector)
        print("Corrected Fourier-space lattice vectors:")
        for v in corrected_basis_vectors:
            print(v)

    # 从傅里叶空间晶格确定实空间晶格
    area = np.cross(corrected_basis_vectors[0], corrected_basis_vectors[1])  # 平行四边形面积
    rotate_90 = ((0., -1.), (1., 0.))  # 逆时针旋转90度的旋转矩阵
    direct_lattice_vectors = [np.dot(v, rotate_90) * fft_abs.shape / area for v in corrected_basis_vectors]
    if verbose:
        print("Real-space lattice vectors:")
        for v in direct_lattice_vectors:
            print(v, "(Magnitude", np.sqrt((v ** 2).sum()), ")")
        print("Lattice vector triangle sum:")
        print(sum(direct_lattice_vectors))
        print("Unit cell area: (%0.2f)^2 square pixels" % (
            np.sqrt(np.abs(np.cross(direct_lattice_vectors[0], direct_lattice_vectors[1])))))

    # # 看一下实空间中的基向量长什么样
    # if display:
    #     show_lattice_overlay1(calibration_all, direct_lattice_vectors, verbose=verbose)

    # 使用实空间中晶格向量和图像数据来测量（第一张校准图像）偏移向量
    offset_vector = get_offset_vector(
        image=calibration_all[0, :, :],
        direct_lattice_vectors=direct_lattice_vectors,
        verbose=verbose, display=display,
        show_interpolation=show_interpolation)

    shift_vector = get_shift_vector(
        corrected_basis_vectors, fft_data_folder, filtered_fft_abs,
        num_harmonics=num_harmonics, outlier_phase=outlier_phase,
        verbose=verbose, display=display,
        scan_type=scan_type, scan_dimensions=scan_dimensions)

    corrected_shift_vector, final_offset_vector = get_precise_shift_vector(
        direct_lattice_vectors, shift_vector, offset_vector,
        calibration_all[-1, :, :], zPix, scan_type, verbose)

    if show_lattice:
        show_lattice_overlay_all(calibration_all, direct_lattice_vectors, offset_vector, corrected_shift_vector,
                                 dot_size=dot_size_show)

        # which_filename = 0
        # while True:
        #     print("Displaying:", filename_list[which_filename])
        #     image_data = load_image_data(filename_list[which_filename])
        #     show_lattice_overlay_all(
        #         image_data, direct_lattice_vectors,
        #         offset_vector, corrected_shift_vector)
        #     if len(filename_list) > 1:
        #         which_filename = input(
        #             "Display lattice overlay for which dataset? [done]:")
        #         try:
        #             which_filename = int(which_filename)
        #         except ValueError:
        #             if which_filename == '':
        #                 print("Done displaying lattice overlay.")
        #                 break
        #             else:
        #                 continue
        #         if which_filename >= len(filename_list):
        #             which_filename = len(filename_list) - 1
        #     else:
        #         break

    # 关闭所有的 matplotlib 图形窗口，并且调用 Python 的垃圾回收机制来释放不再使用的内存。
    if display or show_lattice:
        plt.close('all')
        import gc
        gc.collect()

    if record_parameters:
        params_file_path = os.path.join(result_path, 'parameters.txt')

        with open(params_file_path, 'w') as params:
            params.write("Direct lattice vectors: {}\n\n".format(repr(direct_lattice_vectors)))
            params.write("Corrected shift vector: {}\n\n".format(repr(corrected_shift_vector)))
            params.write("Offset vector: {}\n\n".format(repr(offset_vector)))
            try:
                params.write("Final offset vector: {}\n\n".format(repr(final_offset_vector)))
            except UnboundLocalError:
                params.write("Final offset vector: Not recorded\n\n")
            if calibration is not None:
                params.write("Calibration filename: {}\n\n".format(calibration))

    if calibration is None or background is None:
        return calibration_all.shape, direct_lattice_vectors, corrected_shift_vector, offset_vector
    else:
        # 校准图像光斑强度
        intensities_vs_galvo_position, background_frame = spot_intensity_position(calibration, background,
                                                                                  xPix, yPix,
                                                                                  direct_lattice_vectors,
                                                                                  corrected_shift_vector,
                                                                                  offset_vector,
                                                                                  window_size=calibration_window_size,
                                                                                  show_steps=show_calibration_steps,
                                                                                  display=False,
                                                                                  verbose=verbose,
                                                                                  result_path=result_path)
        return calibration_all.shape, direct_lattice_vectors, corrected_shift_vector, offset_vector, intensities_vs_galvo_position, background_frame


def reconstruct_image_parallel(
        data_filename, calibration_filename,
        xPix, yPix, zPix, steps,
        lattice_vectors, offset_vector, shift_vector,
        new_grid_xrange, new_grid_yrange,
        background_filename = None,
        result_path=None,
        num_processes=1,
        window_footprint=10,
        aperture_size=3,
        make_widefield_image=True,
        make_confocal_image=False,  # Broken, for now
        verbose=True,
        show_steps=False,  # For debugging
        show_slices=False,  # For debugging
        intermediate_data=False,  # Memory hog, for stupid reasons, leave 'False'
        normalize=False,  # Of uncertain merit, leave 'False' probably
        display=False,
        cover=False,
):
    input_arguments = locals()  # 收集所有局部变量
    input_arguments.pop('num_processes')

    if result_path is None:
        filename = os.path.splitext(os.path.basename(calibration_filename))[0]
        # timestamp = datetime.now().strftime("_%Y%m%d_%H_%M_%S")
        timestamp = datetime.now().strftime("_%Y%m%d")
        result_path = os.path.join(os.path.join(os.getcwd(), "result"), filename + timestamp)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if verbose:
            print(f"result_path: {result_path}\n")

    basename = os.path.splitext(os.path.basename(data_filename))[0]  # 去掉文件扩展名，只取文件名部分
    reconstruct_image_name = os.path.join(result_path, basename + '_reconstruct.tif')

    print(f"\nReconstruct {basename}...")
    print(reconstruct_image_name)

    if not cover and os.path.exists(reconstruct_image_name):
        print("\nImage already reconstructed.")
        print("Loading", os.path.split(enderlein_image_name)[1])
        images = {}
        try:
            images['reconstruct_image'] = load_image_data(reconstruct_image_name)
        except ValueError:
            print("\n\nWARNING: the data file:")
            print(enderlein_image_name)
            print("may not be the size it was expected to be.\n\n")
            raise
    else:
        start_time = time.perf_counter()
        if num_processes == 1:
            images = enderlein_image_subprocess(**input_arguments)
#         else:
#             input_arguments['intermediate_data'] = False  # Difficult for parallel
#             input_arguments['show_steps'] = False  # Difficult for parallel
#             input_arguments['show_slices'] = False  # Difficult for parallel
#             input_arguments['display'] = False  # Annoying for parallel
#             input_arguments['verbose'] = False  # Annoying for parallel
#
#             step_boundaries = list(range(0, steps, 10)) + [steps]
#             step_boundaries = [(step_boundaries[i], step_boundaries[i + 1] - 1) for i in
#                                range(len(step_boundaries) - 1)]
#             running_processes = {}
#             first_harvest = True
#             random_prefix = '%06i_' % (random.randint(0, 999999))
#             while len(running_processes) > 0 or len(step_boundaries) > 0:
#                 """Load up the subprocesses"""
#                 while (len(running_processes) < num_processes and
#                        len(step_boundaries) > 0):
#                     sb = step_boundaries.pop(0)
#                     input_arguments['start_frame'], input_arguments['end_frame'] = sb
#                     output_filename = (random_prefix + '%i_%i_intermediate_data.temp' % sb)
#                     sys.stdout.write("\rProcessing frames: " + repr(sb[0]) + '-' + repr(sb[1]) + ' ' * 10)
#                     sys.stdout.flush()
#                     command_string = """
# import array_illumination, pickle
# from numpy import array
# input_arguments=%s
# sub_images = array_illumination.enderlein_image_subprocess(**input_arguments)
# pickle.dump(sub_images, open('%s', 'wb'), protocol=2)
# """ % (repr(input_arguments), output_filename)
#                     running_processes[output_filename] = subprocess.Popen(
#                         [sys.executable, '-c %s' % command_string],
#                         stdin=subprocess.PIPE,
#                         stdout=subprocess.PIPE,
#                         stderr=subprocess.PIPE)
#                 """Poke each subprocess, harvest the finished ones"""
#                 pop_me = []
#                 for f, proc in running_processes.items():
#                     if proc.poll() is not None:  # Time to harvest
#                         pop_me.append(f)
#                         report = proc.communicate()
#                         if report != (b'', b''):
#                             print(report)
#                             raise UserWarning("Problem with a subprocess.")
#                         sub_images = pickle.load(open(f, 'rb'))
#                         os.remove(f)
#
#                         if first_harvest:
#                             images = sub_images
#                             first_harvest = False
#                         else:
#                             for k in images.keys():
#                                 images[k] += sub_images[k]
#                 for p in pop_me:  # Forget about the harvested processes
#                     running_processes.pop(p)
#                 """Chill for a second"""
#                 time.sleep(0.2)
#         end_time = time.perf_counter()
#         print("Elapsed time: %0.2f seconds" % (end_time - start_time))
#         images['enderlein_image'].tofile(enderlein_image_name)
#         if make_widefield_image:
#             images['widefield_image'].tofile(basename + '_widefield.raw')
#         if make_confocal_image:
#             images['confocal_image'].tofile(basename + '_confocal.raw')
#     display = True
#     if display:
#         plt.figure()
#         plt.imshow(images['enderlein_image'], interpolation='nearest', cmap="gray")
#         plt.colorbar()
#         plt.show()
#     return images

def enderlein_image_subprocess(
        data_filename, lake_filename, background_filename,
        xPix, yPix, zPix, steps, preframes,
        lattice_vectors, offset_vector, shift_vector,
        new_grid_xrange, new_grid_yrange,
        start_frame=None, end_frame=None,
        window_footprint=10,
        aperture_size=3,
        make_widefield_image=True,
        make_confocal_image=False,  # Broken, for now
        verbose=True,
        show_steps=False,  # For debugging
        show_slices=False,  # For debugging
        intermediate_data=False,  # Memory hog, for stupid reasons. Leave 'False'
        normalize=False,  # Of uncertain merit, leave 'False' probably
        display=False,
):
    basename = os.path.splitext(data_filename)[0]
#     enderlein_image_name = basename + '_enderlein_image.raw'
#     lake_basename = os.path.splitext(lake_filename)[0]
#     lake_intensities_name = lake_basename + '_spot_intensities.pkl'
#     background_basename = os.path.splitext(background_filename)[0]
#     background_name = background_basename + '_background_image.raw'
#
#     intensities_vs_galvo_position = pickle.load(open(lake_intensities_name, 'rb'))
#     background_directory_name = os.path.dirname(background_name)
#     try:
#         background_frame = np.fromfile(background_name).reshape(xPix, yPix).astype(float)
#     except ValueError:
#         print("\n\nWARNING: the data file:")
#         print(background_name)
#         print("may not be the size it was expected to be.\n\n")
#         raise
#     try:
#         hot_pixels = np.fromfile(os.path.join(background_directory_name, 'hot_pixels.txt'), sep=', ')
#     except:
#         hot_pixels = None
#
#     else:
#         hot_pixels = hot_pixels.reshape(2, len(hot_pixels) // 2)
#
#     if show_steps or show_slices: fig = plt.figure()
#     if start_frame is None:
#         start_frame = 0
#     if end_frame is None:
#         end_frame = steps - 1
#     new_grid_x = np.linspace(*new_grid_xrange)
#     new_grid_y = np.linspace(*new_grid_yrange)
#     enderlein_image = np.zeros((new_grid_x.shape[0], new_grid_y.shape[0]), dtype=np.float64)
#     enderlein_normalization = np.zeros_like(enderlein_image)
#     this_frames_enderlein_image = np.zeros_like(enderlein_image)
#     this_frames_normalization = np.zeros_like(enderlein_image)
#     if intermediate_data:
#         cumulative_sum = np.memmap(
#             basename + '_cumsum.raw', dtype=float, mode='w+',
#             shape=(steps,) + enderlein_image.shape)
#         processed_frames = np.memmap(
#             basename + '_frames.raw', dtype=float, mode='w+',
#             shape=(steps,) + enderlein_image.shape)
#     if make_widefield_image:
#         widefield_image = np.zeros_like(enderlein_image)
#         widefield_coordinates = np.meshgrid(new_grid_x, new_grid_y)
#         widefield_coordinates = (
#             widefield_coordinates[0].reshape(
#                 new_grid_x.shape[0] * new_grid_y.shape[0]),
#             widefield_coordinates[1].reshape(
#                 new_grid_x.shape[0] * new_grid_y.shape[0]))
#     if make_confocal_image:
#         confocal_image = np.zeros_like(enderlein_image)
#     enderlein_normalization.fill(1e-12)
#     aperture = gaussian(2 * window_footprint + 1, std=aperture_size
#                         ).reshape(2 * window_footprint + 1, 1)
#     aperture = aperture * aperture.T
#     grid_step_x = new_grid_x[1] - new_grid_x[0]
#     grid_step_y = new_grid_y[1] - new_grid_y[0]
#     subgrid_footprint = np.floor(
#         (-1 + window_footprint * 0.5 / grid_step_x,
#          -1 + window_footprint * 0.5 / grid_step_y))
#     subgrid = (  # Add 2*(r_0 - r_M) to this to get s_desired
#         window_footprint + 2 * grid_step_x * np.arange(
#             -subgrid_footprint[0], subgrid_footprint[0] + 1),
#         window_footprint + 2 * grid_step_y * np.arange(
#             -subgrid_footprint[1], subgrid_footprint[1] + 1))
#     subgrid_points = ((2 * subgrid_footprint[0] + 1) *
#                       (2 * subgrid_footprint[1] + 1))
#     for z in range(start_frame, end_frame + 1):
#         im = load_image_slice(
#             filename=data_filename, xPix=xPix, yPix=yPix,
#             preframes=preframes, which_slice=z).astype(float)
#         if hot_pixels is not None:
#             im = remove_hot_pixels(im, hot_pixels)
#         this_frames_enderlein_image.fill(0.)
#         this_frames_normalization.fill(1e-12)
#         if verbose:
#             sys.stdout.write("\rProcessing raw data image %i" % (z))
#             sys.stdout.flush()
#         if make_widefield_image:
#             widefield_image += interpolation.map_coordinates(im, widefield_coordinates).reshape(new_grid_y.shape[0],
#                                                                                                 new_grid_x.shape[0]).T
#         lattice_points, i_list, j_list = (generate_lattice(image_shape=(xPix, yPix), lattice_vectors=lattice_vectors,
#                                                            center_pix=offset_vector + get_shift(shift_vector, z),
#                                                            edge_buffer=window_footprint + 1, return_i_j=True))
#         for m, lp in enumerate(lattice_points):
#             i, j = int(i_list[m]), int(j_list[m])
#             """Take an image centered on each illumination point"""
#             spot_image = get_centered_subimage(center_point=lp, window_size=window_footprint, image=im,
#                                                background=background_frame)
#             """Aperture the image with a synthetic pinhole"""
#             intensity_normalization = 1.0 / (intensities_vs_galvo_position.get((i, j), {}).get(z, np.inf))
#             if (intensity_normalization == 0 or spot_image.shape != (
#                     2 * window_footprint + 1, 2 * window_footprint + 1)):
#                 continue  # Skip to the next spot
#             apertured_image = (aperture * spot_image * intensity_normalization)
#             nearest_grid_index = np.round((lp - (new_grid_x[0], new_grid_y[0])) / (grid_step_x, grid_step_y))
#             nearest_grid_point = ((new_grid_x[0], new_grid_y[0]) + (grid_step_x, grid_step_y) * nearest_grid_index)
#             new_coordinates = np.meshgrid(subgrid[0] + 2 * (nearest_grid_point[0] - lp[0]),
#                                           subgrid[1] + 2 * (nearest_grid_point[1] - lp[1]))
#             resampled_image = interpolation.map_coordinates(apertured_image, (
#                 new_coordinates[0].reshape(int(subgrid_points)),
#                 new_coordinates[1].reshape(int(subgrid_points)))).reshape(int(2 * subgrid_footprint[1] + 1),
#                                                                           int(2 * subgrid_footprint[0] + 1)).T
#             """Add the recentered image back to the scan grid"""
#             if intensity_normalization > 0:
#                 this_frames_enderlein_image[
#                 int(nearest_grid_index[0] - subgrid_footprint[0]):int(nearest_grid_index[0] + subgrid_footprint[0] + 1),
#                 int(nearest_grid_index[1] - subgrid_footprint[1]):int(nearest_grid_index[1] + subgrid_footprint[1] + 1),
#                 ] += resampled_image
#                 this_frames_normalization[
#                 int(nearest_grid_index[0] - subgrid_footprint[0]):int(nearest_grid_index[0] + subgrid_footprint[0] + 1),
#                 int(nearest_grid_index[1] - subgrid_footprint[1]):int(nearest_grid_index[1] + subgrid_footprint[1] + 1),
#                 ] += 1
#                 if make_confocal_image:  # FIXME!!!!!!!
#                     confocal_image[
#                     nearest_grid_index[0] - window_footprint:nearest_grid_index[0] + window_footprint + 1,
#                     nearest_grid_index[1] - window_footprint:nearest_grid_index[1] + window_footprint + 1
#                     ] += interpolation.shift(
#                         apertured_image, shift=(lp - nearest_grid_point))
#             if show_steps:
#                 plt.clf()
#                 plt.suptitle(
#                     "Spot %i, %i in frame %i\nCentered at %0.2f, %0.2f\n" % (i, j, z, lp[0], lp[1]) + (
#                             "Nearest grid point: %i, %i" % (nearest_grid_point[0], nearest_grid_point[1])))
#                 plt.subplot(1, 3, 1)
#                 plt.imshow(
#                     spot_image, interpolation='nearest', cmap="gray")
#                 plt.subplot(1, 3, 2)
#                 plt.imshow(apertured_image, interpolation='nearest', cmap="gray")
#                 plt.subplot(1, 3, 3)
#                 plt.imshow(resampled_image, interpolation='nearest', cmap="gray")
#                 fig.show()
#                 fig.canvas.draw()
#                 response = input('\nHit enter to continue, q to quit:')
#                 if response == 'q' or response == 'e' or response == 'x':
#                     print("Done showing steps...")
#                     show_steps = False
#         enderlein_image += this_frames_enderlein_image
#         enderlein_normalization += this_frames_normalization
#         if not normalize:
#             enderlein_normalization.fill(1)
#             this_frames_normalization.fill(1)
#         if intermediate_data:
#             cumulative_sum[z, :, :] = (enderlein_image * 1. / enderlein_normalization)
#             cumulative_sum.flush()
#             processed_frames[z, :, :] = this_frames_enderlein_image * 1. / (this_frames_normalization)
#             processed_frames.flush()
#         if show_slices:
#             plt.clf()
#             plt.imshow(enderlein_image * 1.0 / enderlein_normalization, cmap="gray", interpolation='nearest')
#             fig.show()
#             fig.canvas.draw()
#             response = input('Hit enter to continue...')
#
#     images = {}
#     images['enderlein_image'] = (enderlein_image * 1.0 / enderlein_normalization)
#     if make_widefield_image:
#         images['widefield_image'] = widefield_image
#     if make_confocal_image:
#         images['confocal_image'] = confocal_image
#     return images


# def show_lattice_overlay1(calibration_all, direct_lattice_vectors, verbose=False):
#     """
#     展示 calibration_all 的第一张图片，并在图片上叠加原点（图像中点）、三个二维向量（从原点出发），叠加的图形用红色展示。
#
#     :param calibration_all: 三维图堆栈
#     :param direct_lattice_vectors: 三个二维向量
#     :param verbose: 是否打印详细信息，默认为 False
#     """
#     # 获取第一张图片
#     first_image = calibration_all[0]
#
#     # 计算图像的中心点
#     center_y, center_x = np.array(first_image.shape) // 2
#
#     # 绘制第一张图片
#     plt.imshow(first_image, cmap='gray')
#
#     # 绘制原点
#     plt.scatter(center_x, center_y, color='red', s=50)
#
#     # 绘制三个二维向量
#     for vector in direct_lattice_vectors:
#         # 按坐标系论的xy，因此array的坐标顺序需要反一下
#         plt.quiver(center_x, center_y, vector[1], vector[0], angles='xy', scale_units='xy', scale=1, color='red')
#
#     # 设置坐标轴比例
#     plt.axis('equal')
#
#     # 显示图形
#     plt.axis('off')
#     plt.show()
#
#     if verbose:
#         print("Lattice overlay displayed successfully.")


def show_lattice_overlay_origin(calibration_all, direct_lattice_vectors, lattice_points, offset_vector=None,
                                verbose=False):
    """
    展示 calibration_all 的第一张图片，并在图片上叠加原点（图像中点）、三个二维向量（从原点出发）、晶格点，叠加的图形用红色展示。

    :param calibration_all: 三维图堆栈，这里可以传入单张二维图当作特殊的三维图堆栈
    :param direct_lattice_vectors: 三个二维向量
    :param lattice_points: 晶格点
    :param offset_vector: 偏移向量
    :param verbose: 是否打印详细信息，默认为 False
    """
    # 获取第一张图片
    first_image = calibration_all if calibration_all.ndim == 2 else calibration_all[0]

    # 计算图像的中心点
    if offset_vector is None:
        center_y, center_x = np.array(first_image.shape) // 2
    else:
        center_y, center_x = offset_vector

    # 绘制第一张图片
    plt.imshow(first_image, cmap='gray')

    # 绘制晶格点
    plt.scatter(np.array(lattice_points)[:, 1], np.array(lattice_points)[:, 0], color='red', s=10)

    # 绘制原点
    plt.scatter(center_x, center_y, color='red', s=50)

    # 绘制三个二维向量
    for vector in direct_lattice_vectors:
        # 按坐标系论的xy，因此array的坐标顺序需要反一下
        plt.quiver(center_x, center_y, vector[1], vector[0], angles='xy', scale_units='xy', scale=1, color='red')

    # 设置坐标轴比例
    plt.axis('equal')

    # 显示图形
    plt.axis('off')

    if verbose:
        print("Lattice overlay displayed successfully.")


def show_lattice_overlay_all(image_data, direct_lattice_vectors, offset_vector, shift_vector, dot_size=1):
    plt.figure()
    s = 0
    while True:
        plt.clf()
        show_me = median_filter(np.array(image_data[s, :, :]), size=3)
        dots = np.zeros(list(show_me.shape) + [4])
        lattice_points = generate_lattice(
            show_me.shape, direct_lattice_vectors,
            center_pix=offset_vector + get_shift(shift_vector, s))
        radius = dot_size // 2  # 计算半径
        for lp in lattice_points:
            x, y = np.round(lp).astype(int)
            # 根据 dot_size 调整点的大小，这里简单地复制点周围的像素来增大点的视觉效果
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    new_x, new_y = x + i, y + j
                    if 0 <= new_x < show_me.shape[0] and 0 <= new_y < show_me.shape[1]:
                        dots[new_x, new_y, 0::3] = 1

        plt.imshow(show_me, cmap="gray", interpolation='nearest')
        plt.imshow(dots, interpolation='nearest')
        plt.title("Red dots show the calculated illumination pattern")
        plt.show()

        new_s = input("Next frame [exit]:")
        if new_s == '':
            print("Exiting")
            break
        try:
            s = int(new_s)
        except ValueError:
            print("Response not understood. Exiting display.")
            break
        s %= image_data.shape[0]
        print("Displaying frame %i" % (s))

    return None


# def detect_dot_centers(image, weighted=False, verbose=False, show=False):
#     """
#     检测点阵图像中所有点的质心。
#
#     :param image: 输入的灰度图像
#     :param weighted: 是否使用加权质心计算，默认为 False
#     :param verbose: 是否打印详细信息，默认为 False
#     :param show: 是否显示原始图像、二值图像和标记质心的图像，默认为 False
#     :return: 检测到的圆点质心列表
#     """
#     # 图像预处理：阈值处理
#     _, binary_image = cv2.threshold(image, 2 * image.min() / 3. + image.max() / 3., image.max(), cv2.THRESH_BINARY)
#     binary_image = binary_image.astype(np.uint8)  # 转换为uint8
#
#     dot_centers_stack = []
#     for binary_img in binary_image:
#         # 查找轮廓
#         contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         dot_centers = []
#         if ~weighted:
#             for contour in contours:
#                 # 计算轮廓的矩
#                 M = cv2.moments(contour)
#
#                 # 计算质心
#                 if M["m00"] != 0:
#                     cx = round(M["m10"] / M["m00"])
#                     cy = round(M["m01"] / M["m00"])
#                     dot_centers.append((cx, cy))
#         else:
#             for contour in contours:
#                 M = cv2.moments(contour)
#                 center = None
#                 if M["m00"] != 0:
#                     center = weighted_centroid(image, contour)
#                 if center is not None:
#                     dot_centers.append(center)
#         print("Detected %d dots centers" % len(dot_centers))
#         dot_centers_stack.append(dot_centers)
#
#     if show:
#         for index, dot_centers in enumerate(dot_centers_stack):
#             plt.figure(figsize=(16, 8))
#             plt.subplot(131)
#             plt.imshow(image[index], cmap='gray')
#             plt.title("Original Image")
#
#             plt.subplot(132)
#             plt.imshow(binary_image[index], cmap='gray')
#             plt.title("Binary Image")
#
#             color_image = cv2.cvtColor(image[index], cv2.COLOR_GRAY2BGR)
#             color_image = array_scale(color_image)
#             for center in dot_centers:
#                 cv2.circle(color_image, center, 0, (0, 0, 255), -1)  # 在质心位置画一个红色圆点
#
#             plt.subplot(133)
#             plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
#             if ~weighted:
#                 plt.title("Detected Unweighted Centers")
#             else:
#                 plt.title("Detected Weighted Centers")
#
#             plt.show()
#             flag = input("Continue?[y]/n: ")
#             # break
#
#     if verbose:
#         for i, center in enumerate(dot_centers):
#             print(f"Dot {i + 1} weighted center: ({center[0]}, {center[1]})")
#
#     return dot_centers


def get_fft_abs(filename, image_data, result_path, verbose=False, show_steps=False):
    """
    计算图像数据的傅里叶变换，并返回FFT的绝对值和平均值。
    如果之前已经对相同文件进行过FFT计算且结果文件存在，函数将直接加载这些结果，避免重复计算。

    :param filename: 输入图像文件名
    :param image_data: 输入图像数据
    :param show_steps: 是否显示每一步的结果，默认为 False
    :return: fft_data_folder, fft_abs, fft_avg
    """

    # 快速傅里叶变换（FFT）数据以一系列原始二进制文件的形式存储，每个二维z切片对应一个文件。这些文件的命名为000000.dat、000001.dat...
    basename = os.path.splitext(os.path.basename(filename))[0]  # 去掉文件扩展名，只取文件名部分
    fft_abs_name = os.path.join(result_path, basename + '_fft_abs.npy')
    fft_avg_name = os.path.join(result_path, basename + '_fft_avg.npy')
    fft_data_folder = os.path.join(result_path, basename + '_fft_data')

    # 检查之前是否已经计算过相同文件的FFT结果，若存在则直接加载
    if (os.path.exists(fft_abs_name) and
            os.path.exists(fft_avg_name) and
            os.path.exists(fft_data_folder)):
        print("FFT already calculated.")
        if verbose:
            print("Loading", os.path.split(fft_abs_name)[1])
        fft_abs = np.load(fft_abs_name)
        if verbose:
            print("Loading", os.path.split(fft_avg_name)[1])
        fft_avg = np.load(fft_avg_name)
    else:
        print("Generating fft_abs, fft_avg and fft_data...")
        if not os.path.exists(fft_data_folder):
            os.mkdir(fft_data_folder)
        fft_abs = np.zeros(image_data.shape[1:])
        fft_avg = np.zeros(image_data.shape[1:], dtype=np.complex128)
        window = (hann(image_data.shape[1]).reshape(image_data.shape[1], 1) *
                  hann(image_data.shape[2]).reshape(1, image_data.shape[2]))  # Multiplication of matrices
        if show_steps:
            fig = plt.figure()
        for z in range(image_data.shape[0]):
            fft_data = np.fft.fftshift(  # Stored shifted!
                np.fft.fftn(window * image_data[z, :, :], axes=(0, 1)))
            fft_data.tofile(os.path.join(fft_data_folder, '%06i.dat' % (z)))
            fft_abs += np.abs(fft_data)
            if show_steps:
                plt.clf()
                plt.subplot(1, 3, 1)
                plt.title('Windowed slice %i' % z)
                plt.imshow(window * np.array(image_data[z, :, :]), cmap="gray", interpolation='nearest')
                plt.subplot(1, 3, 2)
                plt.title('FFT of slice %i' % z)
                plt.imshow(np.log(1 + np.abs(fft_data)), cmap="gray", interpolation='nearest')
                plt.subplot(1, 3, 3)
                plt.title("Cumulative sum of FFT absolute values")
                plt.imshow(np.log(1 + fft_abs), cmap="gray", interpolation='nearest')
                plt.show()
                input("Hit enter to continue...")
            fft_avg += fft_data
            sys.stdout.write('\rFourier transforming slice %i' % (z + 1))
            sys.stdout.flush()
        fft_avg = np.abs(fft_avg)
        np.save(fft_abs_name, fft_abs)
        np.save(fft_avg_name, fft_avg)

    return fft_data_folder, fft_abs, fft_avg


def spike_filter(fft_abs, display=False):
    """
    对傅里叶变换的绝对值进行滤波，以减少噪声并突出主要的峰值。
    :param fft_abs:
    :param display:
    :return:
    """
    # 高斯滤波，平滑处理
    f = gaussian_filter(np.log(1 + fft_abs), sigma=0.5)
    if display:
        display_image(f, 'Smoothed')

    # 水平方向滤波
    f = f - gaussian_filter(f, sigma=(0, 4))
    if display:
        display_image(f, 'Filtered left-right')

    # 垂直方向滤波
    f = f - gaussian_filter(f, sigma=(4, 0))
    if display:
        display_image(f, 'Filtered up-down')

    # 再次平滑处理
    f = gaussian_filter(f, sigma=1.5)
    if display:
        display_image(f, 'Resmoothed')

    # 截断
    f = f * (f > 0)
    if display:
        display_image(f, 'Negative truncated')

    # 标准化处理
    f -= f.mean()
    f *= 1.0 / f.std()
    return f


def display_image(f, title):
    """
    显示图像并等待用户输入
    :param f: 要显示的图像数据
    :param title: 图像的标题
    """
    plt.imshow(f, cmap="gray", interpolation='nearest')
    plt.title(title)
    plt.show()


def find_spikes(fft_abs, filtered_fft_abs, extent=15, num_spikes=300, low_pass_filter=0.5, show_ratio=1., display=True,
                animate=False):
    """
    查找傅里叶变换的绝对值中最大的峰值，这些峰值通常对应于图像中的亮点。
    :param fft_abs: 傅里叶变换的绝对值之和
    :param filtered_fft_abs: 滤波后的傅里叶变换的绝对值之和
    :param extent: 峰值的搜索范围
    :param num_spikes: 查找峰值的最大次数
    :param display: 是否显示fft_abs 和 filtered_fft_abs 的图像
    :param animate: 是否显示查找过程
    :return:
    """
    center_pix = np.array(fft_abs.shape) // 2
    log_fft_abs = np.log(1 + fft_abs)
    filtered_fft_abs = np.array(filtered_fft_abs)

    # 近似低通滤波器
    mask1 = int(low_pass_filter * filtered_fft_abs.shape[0] / 2)
    mask2 = int(low_pass_filter * filtered_fft_abs.shape[1] / 2)
    if mask1 > 0 and mask2 > 0:
        filtered_fft_abs[0:mask1, :] = 0
        filtered_fft_abs[-mask1:, :] = 0
        filtered_fft_abs[:, 0:mask2] = 0
        filtered_fft_abs[:, -mask2:] = 0

    if display:
        # 截取fft_abs和 filtered_fft_abs的中心区域
        log_fft_abs_show = log_fft_abs[int(center_pix[0] - center_pix[0] * show_ratio): int(
            center_pix[0] + center_pix[0] * show_ratio),
                           int(center_pix[1] - center_pix[1] * show_ratio): int(
                               center_pix[1] + center_pix[1] * show_ratio)]
        filtered_fft_abs_show = filtered_fft_abs[int(center_pix[0] - center_pix[0] * show_ratio):int(
            center_pix[0] + center_pix[0] * show_ratio),
                                int(center_pix[1] - center_pix[1] * show_ratio): int(
                                    center_pix[1] + center_pix[1] * show_ratio)]

        # 显示 fft_abs 和 filtered_fft_abs 的图像
        image_extent = np.float64([-0.5 - show_ratio * center_pix[1],
                                   filtered_fft_abs.shape[1] - 0.5 - (2 - show_ratio) * center_pix[1],
                                   filtered_fft_abs.shape[0] - 0.5 - (2 - show_ratio) * center_pix[0],
                                   -0.5 - show_ratio * center_pix[0]])  # 左边界、右边界、下边界、上边界（以图像中心为原点）（只是数值，不会截取图像）
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(log_fft_abs_show, cmap="gray", interpolation='nearest', extent=image_extent)
        plt.title('Average Fourier magnitude')
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(filtered_fft_abs_show), cmap="gray", interpolation='nearest', extent=image_extent)
        plt.title('Filtered average Fourier magnitude')
        plt.show()

    coords = []  # 储存尖峰的坐标
    if animate:
        plt.figure()
        print('Center pixel:', center_pix)
    for i in range(num_spikes):
        # print(np.array(np.unravel_index(filtered_fft_abs.argmax(), filtered_fft_abs.shape)), filtered_fft_abs.max())
        # cv2.imwrite("filtered_fft_abs.png", filtered_fft_abs * 255 / filtered_fft_abs.max())
        coords.append(np.array(np.unravel_index(filtered_fft_abs.argmax(), filtered_fft_abs.shape)))
        c = coords[-1]
        # 将当前尖峰周围的区域置为0，避免重复检测
        xSl = slice(max(c[0] - extent, 0), min(c[0] + extent, filtered_fft_abs.shape[0]))
        ySl = slice(max(c[1] - extent, 0), min(c[1] + extent, filtered_fft_abs.shape[1]))

        filtered_fft_abs[xSl, ySl] = 0

        if animate:
            # 截取filtered_fft_abs的中心区域
            filtered_fft_abs_show = filtered_fft_abs[int(center_pix[0] - center_pix[0] * show_ratio):int(
                center_pix[0] + center_pix[0] * show_ratio),
                                    int(center_pix[1] - center_pix[1] * show_ratio): int(
                                        center_pix[1] + center_pix[1] * show_ratio)]

            image_extent = np.float64([0, filtered_fft_abs.shape[1] * show_ratio,
                                       filtered_fft_abs.shape[0] * show_ratio, 0])
            # 左边界、右边界、下边界、上边界（以图像中心为原点）（只是数值，不会截取图像）

            print(i, ':', c)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(filtered_fft_abs_show, cmap="gray", interpolation='nearest', extent=image_extent)
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.plot(filtered_fft_abs_show.max(axis=1))
            plt.show()
            if i == 0:
                input('.')

    coords = [c - center_pix for c in coords]  # 将所有尖峰的坐标转换为相对于图像中心的坐标
    coords = sorted(coords, key=lambda x: x[0] ** 2 + x[1] ** 2)  # 按向量幅度从小到大排序尖峰的坐标

    return coords


def get_basis_vectors(fft_abs, coords, tolerance=3., num_harmonics=3, verbose=False):
    """
    从傅里叶变换的绝对值中找到一组基本向量，这些向量可能对应于图像中的晶格。
    :param fft_abs: 傅里叶变换的绝对值
    :param coords: 尖峰的坐标
    :param tolerance: 查找晶格点时允许的误差容限
    :param num_harmonics: 所需的谐波数量，用于判断是否找到足够的晶格点
    :param verbose: 是否打印详细的调试信息
    :return: 晶格基向量
    """
    for i in range(len(coords)):  # Where to start looking.
        basis_vectors = []
        for c, coord in enumerate(coords):
            if c < i:
                continue

            # 中心峰值
            if c == 0:
                if max(abs(coord)) > 0:  # 第一个最大值不在中心
                    print("c:", c)
                    print("Coord:", coord)
                    print("Coordinates:")
                    for x in coords:
                        print(x)
                    raise UserWarning('No peak at the central pixel')
                else:
                    continue

            if coord[0] < 0 or (coord[0] == 0 and coord[1] < 0):
                # Ignore the negative versions
                if verbose:
                    print("\nIgnoring:", coord)
            else:
                # Check for harmonics
                if verbose:
                    print("\nTesting:", coord)
                num_vectors, points_found = test_basis(coords, [coord], tolerance=tolerance, verbose=verbose)
                if num_vectors > num_harmonics:
                    # 找到了足够的谐波，目前先保留它
                    basis_vectors.append(coord)
                    # center_pix = np.array(fft_abs.shape) // 2
                    # furthest_spike = points_found[-1] + center_pix
                    if verbose:
                        print("Appending", coord)
                        print("%i harmonics found, at:" % (num_vectors - 1))
                        for p in points_found:
                            print(' ', p)

                    # 如果向量单独测试通过了，就需要通过组合测试
                    if len(basis_vectors) > 1:
                        if verbose:
                            print("\nTesting combinations:", basis_vectors)
                        num_vectors, points_found = test_basis(coords, basis_vectors, tolerance=tolerance,
                                                               verbose=verbose)
                        if num_vectors > num_harmonics:
                            # 找到了足够的谐波，组合通过测试
                            if len(basis_vectors) == 3:
                                # 找到三个基向量，则完成任务，查找更准确的基向量
                                precise_basis_vectors = get_precise_basis(coords, basis_vectors, fft_abs,
                                                                          tolerance=tolerance, verbose=verbose)
                                (x_1, x_2, x_3) = sorted(precise_basis_vectors, key=lambda x: abs(x[0]))  # 按元素绝对值大小排序
                                possibilities = sorted(
                                    ([x_1, x_2, x_3],
                                     [x_1, x_2, -x_3],
                                     [x_1, -x_2, x_3],
                                     [x_1, -x_2, -x_3]),
                                    key=lambda x: (np.array(sum(x)) ** 2).sum()
                                )  # 最终结果从小到大排序

                                if verbose:
                                    print("Possible triangle combinations:")
                                    for p in possibilities:
                                        print(" ", p)

                                precise_basis_vectors = possibilities[0]  # 取排序最小的那个
                                if precise_basis_vectors[-1][0] < 0:
                                    for p in range(3):
                                        precise_basis_vectors[p] *= -1
                                return precise_basis_vectors

                        # 组合测试未通过，删除最后进来的向量
                        else:
                            # Blame the new guy, for now.
                            basis_vectors.pop()
    else:
        raise UserWarning(
            "Basis vector search failed. Diagnose by running with verbose=True")


def test_basis(coords, basis_vectors, tolerance, verbose=False):
    """
    查找预期的晶格，返回找到的点，并在失败时停止搜索
    :param coords: 傅里叶图像中的峰值坐标相对于中心原点的向量
    :param basis_vectors: [coord]，或多个通过测试的向量的集合
    :param tolerance:
    :param verbose:
    :return:
    """
    points_found = list(basis_vectors)
    num_vectors = 2
    searching = True
    while searching:
        # 生成基向量的所有可能的组合（允许重复），并将这些组合的和作为预期的晶格点
        if verbose:
            print("Looking for combinations of %i basis vectors." % num_vectors)
        lattice = [sum(c) for c in combinations_with_replacement(basis_vectors, num_vectors)]
        if verbose:
            print("Expected lattice points:", lattice)

        for i, lat in enumerate(lattice):
            for c in coords:
                dif = np.sqrt(((lat - c) ** 2).sum())
                if dif < tolerance:
                    if verbose:
                        print("Found lattice point:", c)
                        print(" Distance:", dif)
                        if len(basis_vectors) == 1:
                            print(" Fundamental:", c * 1.0 / num_vectors)
                    points_found.append(c)
                    break
            else:  # 如果没有找到预期的晶格点，停止搜索
                if verbose:
                    print("Expected lattice point not found")
                searching = False
        if not searching:
            return num_vectors, points_found
        # 增加基向量的组合数量，继续下一轮搜索
        num_vectors += 1


def get_precise_basis(coords, basis_vectors, fft_abs, tolerance, verbose=False):
    """
    使用预期的晶格来估计基向量的精确值。

    :param coords: 尖峰的坐标列表
    :param basis_vectors: 初步的基向量列表
    :param fft_abs: 傅里叶变换的绝对值数组
    :param tolerance: 查找晶格点时允许的误差容限
    :param xPix: 图像的水平像素数，默认为 128
    :param yPix: 图像的垂直像素数，默认为 128
    :param verbose: 是否打印详细的调试信息，默认为 False
    :return: 精确的基向量数组
    """
    if verbose:
        print("\nAdjusting basis vectors to match lattice...")
    center_pix = np.array(fft_abs.shape) // 2
    basis_vectors = list(basis_vectors)
    spike_indices = []  # 晶格点索引
    spike_locations = []  # 晶格点位置

    num_vectors = 2
    searching = True
    while searching:
        # 下面步骤的结果依赖于两次调用函数给出的组合顺序相同
        combinations = [c for c in combinations_with_replacement(basis_vectors, num_vectors)]  # 基向量组和
        combination_indices = [c for c in combinations_with_replacement((0, 1, 2), num_vectors)]  # 索引组合

        for i, comb in enumerate(combinations):
            lat = sum(comb)  # 计算组合的和，得到预期的晶格点
            key = tuple([combination_indices[i].count(v) for v in (0, 1, 2)])  # 计算0、1、2出现的次数

            for c in coords:
                dif = np.sqrt(((lat - c) ** 2).sum())
                if dif < tolerance:  # 寻找对应晶格点
                    true_max = None
                    p = c + center_pix  # 将尖峰坐标转换为图像中的实际坐标
                    # 检查坐标是否在图像范围内
                    if 0 < p[0] < fft_abs.shape[0] and 0 < p[1] < fft_abs.shape[0]:
                        # 使用 simple_max_finder 函数估计精确的最大值位置（用在空域上是不是可以改为周围一定范围内的质心）
                        true_max = c + simple_max_finder(fft_abs[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2],
                                                         show_plots=False)
                    if verbose:
                        print("Found lattice point:", c)
                        print("Estimated position:", true_max)
                        print("Lattice index:", key)

                    spike_indices.append(key)  # 记录晶格点的索引
                    spike_locations.append(true_max)  # 记录晶格点的精确位置
                    break
            else:  # 没有找到预期的晶格点
                if verbose:
                    print("Expected lattice point not found")
                searching = False
        if not searching:  # 根据找到的尖峰，估计基向量
            A = np.array(spike_indices)  # 晶格点索引矩阵
            v = np.array(spike_locations)  # 晶格点位置矩阵
            # 使用最小二乘法求解精确的基向量（Ax=v）
            precise_basis_vectors, residues, rank, s = np.linalg.lstsq(A, v, rcond=None)
            if verbose:
                print("Precise basis vectors:")
                print(precise_basis_vectors)
                print("Residues:", residues)
                print("Rank:", rank)
                print("s:", s)
                print()
            return precise_basis_vectors
        # 增加基向量的组合数量，继续下一轮搜索
        num_vectors += 1


def combinations_with_replacement(iterable, r):
    """
    用于生成可重复的组合。与普通的组合不同，可重复组合允许元素在组合中重复出现。例如，对于集合 ['a', 'b', 'c']，
    选取 2 个元素的可重复组合包括 ('a', 'a')、('a', 'b') 等。
    print([i for i in combinations_with_replacement(['a', 'b', 'c'], 2)])
    [('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')]
    :param iterable: 元素列表
    :param r: 组合个数
    :return:
    """
    """

    """
    pool = tuple(iterable)
    n = len(pool)
    for indices in product(range(n), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


def simple_max_finder(a, show_plots=True):
    """Given a 3x3 array with the maximum pixel in the center,
    estimates the x/y position of the true maximum"""
    true_max = []
    inter_points = np.arange(-1, 2)
    for data in (a[:, 1], a[1, :]):
        my_fit = np.poly1d(np.polyfit(inter_points, data, deg=2))
        true_max.append(-my_fit[1] / (2.0 * my_fit[2]))

    true_max = np.array(true_max)

    if show_plots:
        print("Correction:", true_max)
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(a, interpolation='nearest', cmap="gray")
        plt.axhline(y=1 + true_max[0])
        plt.axvline(x=1 + true_max[1])
        plt.subplot(1, 3, 2)
        plt.plot(a[:, 1])
        plt.axvline(x=1 + true_max[0])
        plt.subplot(1, 3, 3)
        plt.plot(a[1, :])
        plt.axvline(x=1 + true_max[1])
        plt.show()

    return true_max


def get_offset_vector(image, direct_lattice_vectors, prefilter='median', filter_size=3, verbose=True, display=True,
                      show_interpolation=True):
    """
    已知晶格向量，计算一张图片中晶格点的偏移向量
    :param image:
    :param direct_lattice_vectors:
    :param prefilter:
    :param filter_size:
    :param verbose:
    :param display:
    :param show_interpolation:
    :return:
    """
    # 中值滤波
    if prefilter == 'median':
        image = median_filter(image, size=filter_size)

    if verbose:
        print("\nCalculating offset vector...")

    # 窗口大小：晶格向量x\y方向上的最大距离，加上一个缓冲区（2）
    ws = 2 + int(max([abs(v).max() for v in direct_lattice_vectors]))
    if verbose:
        print("Window size:", ws)

    # 按照窗口大小初始化窗口，shape=(2 * ws + 1, 2 * ws + 1)
    window = np.zeros([2 * ws + 1] * 2, dtype=np.float64)
    lattice_points = generate_lattice(image.shape, direct_lattice_vectors, edge_buffer=2 + ws)

    for lp in lattice_points:
        window += get_centered_subimage(center_point=lp, window_size=ws, image=image.astype(float))

    if display:
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        show_lattice_overlay_origin(image, direct_lattice_vectors, lattice_points, verbose=verbose)
        plt.title('Original Lattice Overlay')
        plt.subplot(222)
        plt.imshow(window, interpolation='nearest', cmap="gray")
        plt.title('Original Lattice Average')
        # plt.show()

    # 复制窗口数组，将窗口的上下\左右边界（2）置为 0
    buffered_window = np.array(window)
    buffered_window[:2, :] = 0
    buffered_window[-2:, :] = 0
    buffered_window[:, :2] = 0
    buffered_window[:, -2:] = 0

    while True:  # 查找不在边界的最大值（平均图像中）
        max_pix = np.unravel_index(buffered_window.argmax(), window.shape)
        if (3 < max_pix[0] < window.shape[0] - 3) and (3 < max_pix[1] < window.shape[1] - 3):
            break
        else:
            buffered_window = gaussian_filter(buffered_window, sigma=2)

    if verbose:
        print("Maximum pixel in lattice average:", max_pix)

    correction = simple_max_finder(window[max_pix[0] - 1:max_pix[0] + 2, max_pix[1] - 1:max_pix[1] + 2],
                                   show_plots=show_interpolation)  # 估计最大像素在平均window中的精确位置

    offset_vector = max_pix + correction + np.array(image.shape) // 2 - ws
    if verbose:
        print("Offset vector:", offset_vector)

    # 验证偏移向量是否有效
    window = np.zeros([2 * ws + 1] * 2, dtype=np.float64)
    lattice_points = generate_lattice(image.shape, direct_lattice_vectors, center_pix=offset_vector, edge_buffer=2 + ws)

    if display:
        for lp in lattice_points:
            window += get_centered_subimage(center_point=lp, window_size=ws, image=image.astype(float))
        plt.subplot(223)
        show_lattice_overlay_origin(image, direct_lattice_vectors, lattice_points, offset_vector, verbose=verbose)
        plt.title('Adjusted Lattice Overlay with Offset Vector')
        plt.subplot(224)
        plt.imshow(window, interpolation='nearest', cmap="gray")
        plt.title('Lattice Average\nThis should look like round blobs')
        plt.show()

    return offset_vector


def generate_lattice(image_shape, lattice_vectors, center_pix='image', edge_buffer=2, return_i_j=False):
    """
    根据给定的图像形状、晶格向量和中心像素位置生成晶格点。

    :param image_shape: 图像的形状，通常为一个二元组 (height, width)
    :param lattice_vectors: 晶格向量列表，至少包含两个二维向量
    :param center_pix: 晶格的中心像素位置，可以是字符串 'image' 表示以图像中心为中心，
                       也可以是一个二维数组表示具体的像素坐标，默认为 'image'
    :param edge_buffer: 图像边缘的缓冲区大小，用于过滤掉靠近边缘的晶格点，默认为 2
    :param return_i_j: 是否返回晶格点对应的 i 和 j 索引，默认为 False
    :return: 如果 return_i_j 为 False，返回晶格点列表；
             如果 return_i_j 为 True，返回一个元组，包含晶格点列表、对应的 i 索引列表和 j 索引列表
    """
    if isinstance(center_pix, str):
        if center_pix == 'image':
            # 以图像中心为中心
            center_pix = np.array(image_shape) // 2
    else:
        # 将输入的中心像素位置转换为相对于图像中心的坐标
        center_pix = np.array(center_pix) - (np.array(image_shape) // 2)
        # 求解中心像素在晶格向量下的分量
        lattice_components = np.linalg.solve(np.vstack(lattice_vectors[:2]).T, center_pix)
        # 将分量值限制在 [0, 1) 范围内
        lattice_components_centered = np.mod(lattice_components, 1)
        # 计算分量的整数部分
        lattice_shift = lattice_components - lattice_components_centered
        # 重新计算中心像素位置
        center_pix = lattice_vectors[0] * lattice_components_centered[0] + \
                     lattice_vectors[1] * lattice_components_centered[1] + \
                     np.array(image_shape) // 2

    # 计算生成晶格点所需的向量数量
    num_vectors = int(np.round(1.5 * max(image_shape) / np.sqrt((lattice_vectors[0] ** 2).sum())))  # changed
    # 定义晶格点的范围
    lower_bounds = (edge_buffer, edge_buffer)
    upper_bounds = (image_shape[0] - edge_buffer, image_shape[1] - edge_buffer)
    # 生成二维网格索引
    i, j = np.mgrid[-num_vectors:num_vectors, -num_vectors:num_vectors]
    # 将索引数组展平为一维数组
    i = i.reshape(i.size, 1)
    j = j.reshape(j.size, 1)
    # 根据索引和晶格向量计算晶格点的位置
    lp = i * lattice_vectors[0] + j * lattice_vectors[1] + center_pix
    # 过滤掉超出上下界的晶格点
    valid = np.all(lower_bounds < lp, 1) * np.all(lp < upper_bounds, 1)
    # 提取有效的晶格点
    lattice_points = list(lp[valid])
    if return_i_j:
        # 返回有效的 i\j 索引减去晶格偏移量
        return (lattice_points,
                list(i[valid] - lattice_shift[0]),
                list(j[valid] - lattice_shift[1]))
    else:
        return lattice_points


def get_centered_subimage(
        center_point, window_size, image, background='none'):
    x, y = np.round(center_point).astype(int)
    xSl = slice(max(x - window_size - 1, 0), x + window_size + 2)
    ySl = slice(max(y - window_size - 1, 0), y + window_size + 2)
    subimage = np.array(image[xSl, ySl])

    if not isinstance(background, str):
        subimage -= background[xSl, ySl]
    interpolation.shift(subimage, shift=(x, y) - center_point, output=subimage)
    return subimage[1:-1, 1:-1]


def get_shift_vector(
        fourier_lattice_vectors, fft_data_folder, filtered_fft_abs, num_harmonics=3, outlier_phase=1., verbose=True,
        display=True, scan_type='visitech', scan_dimensions=None):
    if verbose:
        print("\nCalculating shift vector...")

    center_pix = np.array(filtered_fft_abs.shape) // 2
    harmonic_pixels = []
    values = {}
    for v in fourier_lattice_vectors:
        harmonic_pixels.append([])
        for i in range(1, num_harmonics + 1):
            expected_pix = (np.round((i * v)) + center_pix).astype(int)
            roi = filtered_fft_abs[expected_pix[0] - 1:expected_pix[0] + 2, expected_pix[1] - 1:expected_pix[1] + 2]
            shift = -1 + np.array(
                np.unravel_index(roi.argmax(), roi.shape))
            actual_pix = expected_pix + shift - center_pix
            if verbose:
                print("Expected pixel:", expected_pix - center_pix)
                print("Shift:", shift)
                print("Brightest neighboring pixel:", actual_pix)
            harmonic_pixels[-1].append(tuple(actual_pix))
            values[harmonic_pixels[-1][-1]] = []

    num_slices = len(os.listdir(fft_data_folder))
    if verbose:
        print('\n')

    for z in range(num_slices):
        if verbose:
            sys.stdout.write("\rLoading harmonic pixels from FFT slice %06i" % z)
            sys.stdout.flush()
        fft_data = load_fft_slice(fft_data_folder, xPix=filtered_fft_abs.shape[0], yPix=filtered_fft_abs.shape[1],
                                  which_slice=z)
        for hp in harmonic_pixels:
            for p in hp:
                values[p].append(fft_data[p[0] + center_pix[0], p[1] + center_pix[1]])

    if verbose:
        print()

    slopes = []
    k = []
    if display:
        plt.figure()
    if scan_dimensions is not None:
        scan_dimensions = tuple(reversed(scan_dimensions))
    for hp in harmonic_pixels:
        for n, p in enumerate(hp):
            values[p] = np.unwrap(np.angle(values[p]))
            if scan_type == 'visitech':
                slope = np.polyfit(range(len(values[p])), values[p], deg=1)[0]
                values[p] -= slope * np.arange(len(values[p]))
            elif scan_type == 'dmd':
                if scan_dimensions[0] * scan_dimensions[1] != num_slices:
                    raise UserWarning(
                        "The scan dimensions are %i by %i," +
                        " but there are %i slices" % (scan_dimensions[0], scan_dimensions[1], num_slices))
                slope = [0, 0]
                slope[0] = np.polyfit(range(scan_dimensions[1]),
                                      values[p].reshape(scan_dimensions).sum(axis=0) * 1.0 / scan_dimensions[0],
                                      deg=1)[0]
                values[p] -= slope[0] * np.arange(len(values[p]))
                slope[1] = np.polyfit(
                    scan_dimensions[1] * np.arange(scan_dimensions[0]),
                    values[p].reshape(
                        scan_dimensions).sum(axis=1) * 1.0 / scan_dimensions[1],
                    deg=1)[0]
                values[p] -= slope[1] * scan_dimensions[1] * (np.arange(len(values[p])) // scan_dimensions[1])
                slope[1] *= scan_dimensions[1]
            values[p] -= values[p].mean()
            if abs(values[p]).mean() < outlier_phase:
                k.append(p * (-2. * np.pi / np.array(fft_data.shape)))
                slopes.append(slope)
            else:
                if verbose:
                    print("Ignoring outlier:", p)
            if display:
                plt.plot(values[p], '.-', label=repr(p))
    if display:
        plt.title('This should look like noise. Sudden jumps mean bad data!')
        plt.ylabel('Deviation from expected phase')
        plt.xlabel('Image number')
        plt.grid()
        plt.legend(prop={'size': 8})
        plt.axis('tight')
        x_limits = 1.05 * np.array(plt.xlim())
        x_limits -= x_limits[-1] * 0.025
        plt.xlim(x_limits)
        plt.show()

    if scan_type == 'visitech':
        x_s, residues, rank, s = np.linalg.lstsq(np.array(k), np.array(slopes), rcond=None)
    elif scan_type == 'dmd':
        x_s, residues, rank, s = {}, [0, 0], [0, 0], [0, 0]
        x_s['fast_axis'], residues[0], rank[0], s[0] = np.linalg.lstsq(np.array(k), np.array([sl[0] for sl in slopes]),
                                                                       rcond=None)
        x_s['slow_axis'], residues[1], rank[1], s[1] = np.linalg.lstsq(np.array(k), np.array([sl[1] for sl in slopes]),
                                                                       rcond=None)
        x_s['scan_dimensions'] = tuple(reversed(scan_dimensions))

    if verbose:
        print("Shift vector:")
        pprint.pprint(x_s)
        print("Residues:", residues)
        print("Rank:", rank)
        print("s:", s)
    return x_s


def load_fft_slice(fft_data_folder, xPix, yPix, which_slice=0):
    bytes_per_pixel = 16
    filename = os.path.join(fft_data_folder, '%06i.dat' % (which_slice))
    data_file = open(filename, 'rb')
    return np.memmap(data_file, dtype=np.complex128, mode='r').reshape(xPix, yPix)


def get_precise_shift_vector(
        direct_lattice_vectors, shift_vector, offset_vector, last_image, zPix, scan_type, verbose):
    """Use the offset vector to correct the shift vector"""
    final_offset_vector = get_offset_vector(
        image=last_image,
        direct_lattice_vectors=direct_lattice_vectors,
        verbose=False, display=False, show_interpolation=False)

    final_lattice = generate_lattice(last_image.shape, direct_lattice_vectors,
                                     center_pix=offset_vector + get_shift(shift_vector, zPix - 1))
    closest_approach = 1e12
    for p in final_lattice:
        dif = p - final_offset_vector
        distance_sq = (dif ** 2).sum()
        if distance_sq < closest_approach:
            closest_lattice_point = p
            closest_approach = distance_sq
    shift_error = closest_lattice_point - final_offset_vector
    if scan_type == 'visitech':
        movements = zPix - 1
        corrected_shift_vector = shift_vector - (shift_error * 1.0 / movements)
    elif scan_type == 'dmd':
        movements = ((zPix - 1) // shift_vector['scan_dimensions'][0])
        corrected_shift_vector = dict(shift_vector)
        corrected_shift_vector['slow_axis'] = (shift_vector['slow_axis'] - shift_error * 1.0 / movements)

    if verbose:
        print("\nCorrecting shift vector...")
        print(" Initial shift vector:")
        print(' ', pprint.pprint(shift_vector))
        print(" Final offset vector:", final_offset_vector)
        print(" Closest predicted lattice point:", closest_lattice_point)
        print(" Error:", shift_error, "in", movements, "movements")
        print(" Corrected shift vector:")
        print(' ', pprint.pprint(corrected_shift_vector))
        print()

    return corrected_shift_vector, final_offset_vector


def get_shift(shift_vector, frame_number):
    if isinstance(shift_vector, dict):
        """This means we have a 2D shift vector"""
        fast_steps = frame_number % shift_vector['scan_dimensions'][0]
        slow_steps = frame_number // shift_vector['scan_dimensions'][0]
        return (shift_vector['fast_axis'] * fast_steps +
                shift_vector['slow_axis'] * slow_steps)
    else:
        """This means we have a 1D shift vector, like the Visitech Infinity"""
        return frame_number * shift_vector


def spot_intensity_position(calibration_filename, background_filename, xPix, yPix, direct_lattice_vectors,
                            shift_vector, offset_vector, result_path, window_size=5, show_steps=False,
                            display=False, verbose=True):
    """
    使用校准图像堆栈和一组无光照背景图像，校准每个光斑的强度。
    :param calibration_filename: 校准图像堆栈文件的路径
    :param background_filename: 无光照背景图像堆栈文件的路径
    :param xPix: 图像水平像素数
    :param yPix: 图像垂直像素数
    :param direct_lattice_vectors: 晶格向量
    :param shift_vector: 位移向量
    :param offset_vector: 偏移向量
    :param result_path: 结果保存路径
    :param window_size: 校准窗口大小，默认为 5
    :param show_steps: 是否显示每一步的结果，默认为 False
    :param display: 是否显示强度随帧号变化的曲线图，默认为 False
    :return: 一个嵌套字典，记录每个光斑在不同帧的强度；背景图像
    """

    # 生成存储光斑强度的文件名
    calibration_basename = os.path.splitext(os.path.basename(calibration_filename))[0]
    calibration_intensities_name = os.path.join(result_path, calibration_basename + '_spot_intensities.pkl')

    # 获取背景文件名的基础部分，并生成背景结果图像和背景目录的名称
    background_basename = os.path.splitext(os.path.basename(background_filename))[0]
    background_name = os.path.join(result_path, background_basename + '_background_image.tif')
    background_directory_name = os.path.dirname(background_filename)

    # 获取hot_pixels，在背景目录中
    try:
        hot_pixels = np.fromfile(os.path.join(background_directory_name, 'hot_pixels.txt'), sep=', ')
    except IOError:
        # 若热点像素文件不存在，默认继续
        print("\nHot pixel list not found.")
        hot_pixels = None
    else:
        # 成功读取热点像素信息，将其重塑为 2 行的数组
        hot_pixels = hot_pixels.reshape(2, len(hot_pixels) // 2)

    if os.path.exists(calibration_intensities_name) and os.path.exists(background_name):
        # 若光斑强度文件和背景图像文件已存在，直接加载数据
        print("Illumination intensity calibration already calculated.")
        if verbose:
            print("Loading", os.path.split(calibration_intensities_name)[1])
        intensities_position = pickle.load(open(calibration_intensities_name, 'rb'))
        if verbose:
            print("Loading", os.path.split(background_name)[1])
        try:
            # 从文件中读取背景图像数据
            bg = load_image_data(background_name)
        except ValueError:
            print("\n\nWARNING: the data file:")
            print(background_name)
            print("may not be the size it was expected to be.\n\n")
            raise
    else:
        # 若文件不存在，重新计算光斑强度和背景图像
        print("\nCalculating illumination spot intensities...")
        print("Constructing background image...")

        # 加载背景堆栈，计算平均值，然后释放内存
        background_image_data = load_image_data(background_filename)
        bg = np.mean(background_image_data, axis=0)
        del background_image_data

        # 若存在热点像素信息，用3*3窗口中值滤波
        if hot_pixels is not None:
            bg = remove_hot_pixels(bg, hot_pixels)
        print("Background image complete.")

        # 光斑强度字典：元素 [i, j][z] 表示晶格中第 （i,j) 个光斑在第 z 帧中的强度
        calibration_image_data = load_image_data(calibration_filename)
        intensities_position = {}

        if show_steps:
            plt.figure()
        print("Computing flat-field calibration...")
        # 遍历校准图像的每一帧
        for z in range(calibration_image_data.shape[0]):
            im = np.array(calibration_image_data[z, :, :], dtype=float)
            if hot_pixels is not None:
                # 若存在hot_pixel信息，处理
                im = remove_hot_pixels(im, hot_pixels)
            # 打印当前正在处理的校准图像帧号
            sys.stdout.write("\rCalibration image %i" % z)
            sys.stdout.flush()
            # 生成晶格点及其对应的索引
            lattice_points, i_list, j_list = generate_lattice(
                image_shape=(xPix, yPix),
                lattice_vectors=direct_lattice_vectors,
                center_pix=offset_vector + get_shift(shift_vector, z),
                edge_buffer=window_size + 1,
                return_i_j=True)

            # 遍历每个晶格点
            for m, lp in enumerate(lattice_points):
                i, j = int(i_list[m]), int(j_list[m])
                # 获取当前光斑的强度历史记录，若不存在则创建一个新的字典
                intensity_history = intensities_position.setdefault((i, j), {})
                # 获取以当前晶格点为中心的子图像，计算强度之和
                spot_image = get_centered_subimage(center_point=lp, window_size=window_size, image=im, background=bg)
                intensity_history[z] = float(spot_image.sum())
                if show_steps:
                    # 若需要显示每一步的结果，显示当前光斑的子图像
                    plt.clf()
                    plt.imshow(spot_image, interpolation='nearest', cmap="gray")
                    plt.title("Spot %i, %i in frame %i\nCentered at %0.2f, %0.2f" % (i, j, z, lp[0], lp[1]))
                    plt.show()
                    response = input()
                    if response == 'q' or response == 'e' or response == 'x':
                        # 用户选择退出，停止显示每一步的结果
                        print("Done showing steps...")
                        show_steps = False

        # 归一化强度值（打断点，想看看intensities_position长啥样，是不是嵌套字典）
        num_entries = 0
        total_sum = 0
        for hist in intensities_position.values():
            for intensity in hist.values():
                num_entries += 1
                total_sum += intensity
        inverse_avg = num_entries * 1.0 / total_sum
        for hist in intensities_position.values():
            for k in hist.keys():
                hist[k] *= inverse_avg

        # 保存强度字典
        print("\nSaving", os.path.split(calibration_intensities_name)[1])
        pickle.dump(intensities_position, open(calibration_intensities_name, 'wb'), protocol=2)
        print("Saving", os.path.split(background_name)[1])
        utils.save_tiff_2d(background_name, bg)

    # 展示
    if display:
        plt.figure()
        num_lines = 0
        for (i, j), spot_hist in intensities_position.items()[:10]:
            num_lines += 1
            sh = spot_hist.items()
            plt.plot([frame_num for frame_num, junk in sh],
                     [intensity for junk, intensity in sh],
                     ('-', '-.')[num_lines > 5],
                     label=repr((i, j)))
        plt.legend()
        plt.show()

    return intensities_position, bg  # bg is short for 'background'


def load_image_data(filename):
    """
    加载图像三维堆栈
    :param filename:
    :return:
    """
    _, image = cv2.imreadmulti(filename, flags=cv2.IMREAD_UNCHANGED)
    image = np.array(image)

    # 如果三维堆栈是三通道的，转为灰度图
    if len(image.shape) == 4:
        grayscale_all = []
        for i in range(image.shape[0]):
            grayscale_all.append(cv2.cvtColor(image[i], cv2.COLOR_RGB2GRAY))
        image = np.array(grayscale_all)
    return image


def remove_hot_pixels(image, hot_pixels):
    """
    移除图像中的 hot_pixels，用中值滤波，窗口大小3*3
    :param image:
    :param hot_pixels:
    :return:
    """
    height, width = image.shape
    for y, x in hot_pixels:
        x = int(x)
        y = int(y)
        # 处理边界情况
        x_min = max(0, x - 1)
        x_max = min(height, x + 2)
        y_min = max(0, y - 1)
        y_max = min(width, y + 2)
        neighborhood = image[x_min:x_max, y_min:y_max]
        image[x, y] = np.median(neighborhood)
    return image
