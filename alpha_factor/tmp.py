def rolling_jump(sr, win, interval):
    
    if win>=interval:
        result = [np.nan]*(win-interval)
        for i in range(win, len(sr), interval):
            result.extend([np.nan]*(interval-1) + [sr[i-win:i].mean()])
    else:
        result = []
        for i in range(interval, len(sr), interval):
            result.extend([np.nan]*(interval-1) + [sr[i-win:i].mean()])

    print(i, len(result))
    if len(sr) > len(result):
        result.extend([np.nan]*(len(sr) - len(result)))
    return pd.Series(result, sr.index)
