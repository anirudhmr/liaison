mpl_font_size = 14
mpl_linewidth = 3
mpl_params = {
    'axes.labelsize': 20,
    'axes.titlesize': 16,
    'font.size': 24,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.xmargin': .1
}

COLORS = ['red', 'green', 'blue', 'magenta', 'cyan']
# Why not purple?


def set_seaborn_styles(sns):
  # seaborn settings
  # sns.set(font_scale=2)
  # sns.set_context("paper")
  sns.set_style("darkgrid")

  current_palette = sns.color_palette()
  red = current_palette[2]
  blue = current_palette[0]
  green = current_palette[1]
  yellow = current_palette[4]
  current_palette[0] = red
  current_palette[1] = blue
  current_palette[2] = green
  sns.set_palette(current_palette)
  return sns
