using Colors

function rgb(r, g, b)
    return Colors.RGB(r / 255, g / 255, b / 255)
end

uni_tuebingen_primary_colors_by_name = (
    red = rgb(165, 30, 55),
    gold = rgb(180, 160, 105),
    darkgray = rgb(50, 65, 75),
    lightgray = rgb(175, 179, 183),
)

uni_tuebingen_secondary_colors_by_name = (
    darkblue = rgb(65, 90, 140),
    blue = rgb(0, 105, 170),
    lightblue = rgb(80, 170, 200),
    lightgreen = rgb(130, 185, 160),
    green = rgb(125, 165, 75),
    darkgreen = rgb(50, 110, 30),
    ocre = rgb(200, 80, 60),
    violet = rgb(175, 110, 150),
    mauve = rgb(180, 160, 150),
    beige = rgb(215, 180, 105),
    orange = rgb(210, 150, 0),
    brown = rgb(145, 105, 70),
)

uni_tuebingen_colors_by_name =
    merge(uni_tuebingen_primary_colors_by_name, uni_tuebingen_secondary_colors_by_name)

uni_tuebingen_colors = [
    uni_tuebingen_colors_by_name.red,
    uni_tuebingen_colors_by_name.darkgray,
    uni_tuebingen_colors_by_name.gold,
    uni_tuebingen_colors_by_name.lightblue,
    uni_tuebingen_colors_by_name.violet,
    uni_tuebingen_colors_by_name.darkgreen,
    uni_tuebingen_colors_by_name.ocre,
    uni_tuebingen_colors_by_name.brown,
    # uni_tuebingen_colors_by_name.lightgray,
    # uni_tuebingen_colors_by_name.darkblue,
    # uni_tuebingen_colors_by_name.blue,
    # uni_tuebingen_colors_by_name.lightgreen,
    # uni_tuebingen_colors_by_name.green,
    # uni_tuebingen_colors_by_name.mauve,
    # uni_tuebingen_colors_by_name.beige,
    # uni_tuebingen_colors_by_name.orange,
]
