# 自包含的 EEG 预处理函数
import os
import re
import mne

def get_action_label(fname: str) -> str | None:
    """
    如果文件名（不含路径）包含 'LeftAction'（大小写均可），返回 'LA1'；
    否则返回 None。
    """
    if re.search(r'leftaction1', fname, flags=re.I):
        return 'A1L'
    
    if re.search(r'rightaction1', fname, flags=re.I):
        return 'A1R'
    return None

def preprocess_eeg(
    order : str,
    vhdr_file: str,
    save_dir: str = "EEGPP_data",
    locs_file: str = "Standard-10-20-Cap81.locs",
    ref_channels: list[str] | None = None,
    n_components: int = 20,
    l_freq: float = 0.1,
    h_freq: float = 40.0,
    resample_sfreq: int | float = 200,
) -> str:
    """
    读取 BrainVision .vhdr 文件并完成以下预处理:
    1. 加载 Raw 数据
    2. 设置电极定位（montage）
    3. 删除多余通道
    4. 带通滤波
    5. 重新参考
    6. ICA 去伪迹
    7. 降采样
    8. 保存为 .fif
    
    Parameters
    ----------
    order : str
        被试 ID
    vhdr_file : str
        输入 .vhdr 文件路径
    save_dir : str
        预处理后文件保存文件夹
    locs_file : str
        自定义电极定位文件 (.locs)
    ref_channels : list[str] | None
        参考电极通道名列表；None 表示平均参考
    n_components : int
        ICA 分解的组件数
    l_freq : float
        高通截止频率
    h_freq : float
        低通截止频率
    resample_sfreq : int | float
        目标采样率
    
    Returns
    -------
    save_path : str
        保存的 .fif 文件完整路径
    """
    
    file_name = os.path.basename(vhdr_file)
    action_label = get_action_label(file_name)
    print("处理文件: {file_name} -> {action_label}")
    save_name = f"N{order}_{action_label}.fif"
    print(f"保存文件名: {save_name}")
    save_path = os.path.join(save_dir, save_name)
    print(f"保存路径: {save_path}")
    # raise KeyboardInterrupt
    # 读取 BrainVision Raw
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
    
    # 设置电极定位
    montage = mne.channels.read_custom_montage(locs_file, head_size=0.1)
    raw.set_montage(montage, on_missing="warn")
    raw.plot_sensors(show_names=True)

    # 删除不需要的通道
    channels_to_drop = ["eog1", "eog2", "eog3", "eog4", "EMG"]
    raw.drop_channels([ch for ch in channels_to_drop if ch in raw.ch_names])
    
    # 带通滤波
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    # 参考设置
    if ref_channels:
        raw.set_eeg_reference(ref_channels=ref_channels, projection=False)
    else:
        raw.set_eeg_reference("average", projection=True)
    
    # ICA 去伪迹
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter="auto")
    ica.fit(raw)
    # 这里可以手动排除伪迹组件，如需自动化，可自行添加规则
    # ica.exclude = [...]
    ica.apply(raw)
    
    # 重采样
    raw.resample(resample_sfreq)
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    raw.save(save_path, overwrite=True)
    
    print(f"文件 {file_name} 处理完成并保存到: {save_path}")
    return save_path