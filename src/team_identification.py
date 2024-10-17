from src import params

def identify_teams(bounding_boxes: list) -> tuple[list, list]:

    max_x = max(params.VOLLEYBALL_NET[0][0], params.VOLLEYBALL_NET[1][0])
    min_x = min(params.VOLLEYBALL_NET[0][0], params.VOLLEYBALL_NET[1][0])
    mid_x = (max_x + min_x) // 2

    team_1 = []
    team_2 = []

    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box

        center = (int((x + x + w) / 2), int((y + y + h) / 2))

        if center[0] < mid_x:
            team_1.append(bounding_box)
        else:
            team_2.append(bounding_box)

    return team_1, team_2