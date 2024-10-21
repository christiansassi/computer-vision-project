from src import params

def identify_teams(bounding_boxes: list) -> tuple[list, list]:

    max_x = max(params.VOLLEYBALL_NET[0][0], params.VOLLEYBALL_NET[1][0])
    min_x = min(params.VOLLEYBALL_NET[0][0], params.VOLLEYBALL_NET[1][0])
    mid_x = (max_x + min_x) // 2

    team1 = []
    team2 = []

    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box

        center = (int((x + x + w) / 2), int((y + y + h) / 2))

        if center[0] < mid_x:
            team1.append(bounding_box)
        else:
            team2.append(bounding_box)

    return team1, team2