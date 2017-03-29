def speed_heatmap(subset, tracklist):
    speed_diff_mat = pd.DataFrame(data=None, index=subset,
                                  columns=subset)
    for i in subset:
        speed = tracklist[i]['start_spd']
        for j in subset:
            speed2 = tracklist[j]['start_spd']

            speed_diff = np.absolute(np.subtract(speed-speed2))
            speed_diff_mat[i][j] = speed_diff

    speed_diff_mat = speed_diff_mat.apply(pd.to_numeric)

    return speed_diff_mat