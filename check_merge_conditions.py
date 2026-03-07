def check_merge_conditions(raw_eeg, raw_fnirs):
    # 获取基本信息
    eeg_sfreq = raw_eeg.info['sfreq']
    fnirs_sfreq = raw_fnirs.info['sfreq']
    
    eeg_n_times = raw_eeg.n_times
    fnirs_n_times = raw_fnirs.n_times
    
    eeg_start = raw_eeg.times[0]
    fnirs_start = raw_fnirs.times[0]
    
    eeg_stop = raw_eeg.times[-1]
    fnirs_stop = raw_fnirs.times[-1]

    print(f"EEG sfreq: {eeg_sfreq}, fNIRS sfreq: {fnirs_sfreq}")
    print(f"EEG n_times: {eeg_n_times}, fNIRS n_times: {fnirs_n_times}")
    print(f"EEG start: {eeg_start:.3f}s, stop: {eeg_stop:.3f}s")
    print(f"fNIRS start: {fnirs_start:.3f}s, stop: {fnirs_stop:.3f}s")

    # 检查采样率
    sfreq_match = abs(eeg_sfreq - fnirs_sfreq) < 1e-6

    # 检查时间点数量是否相同
    ntimes_match = eeg_n_times == fnirs_n_times

    # 检查起止时间是否一致（可略宽容）
    time_start_match = abs(eeg_start - fnirs_start) < 1e-6
    time_stop_match = abs(eeg_stop - fnirs_stop) < 1e-6

    # 汇总判断
    all_match = sfreq_match and ntimes_match and time_start_match and time_stop_match

    print("\n✅ 可合并" if all_match else "\n❌ 无法直接合并，需对齐")
    
    return {
        'sfreq_match': sfreq_match,
        'n_times_match': ntimes_match,
        'start_match': time_start_match,
        'stop_match': time_stop_match,
        'all_conditions_met': all_match
    }