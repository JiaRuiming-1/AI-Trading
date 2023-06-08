def rolling_jump(sr, win, interval): #@@@
    result = []
    for i in range(win, len(sr), interval):
        if i + interval <= len(sr):
            result.extend([np.nan]*(interval-1) + [sr[i-win:i].mean()])
    if len(sr) > len(result):
        result.extend([np.nan]*(len(sr) % interval))
    return pd.Series(result, sr.index)
