'''
Interactively calculate metrics from slider

TODO !!! Load data from csv
1. It should check for the threshold column as this is used in the slider
2. It should have at least one other column for metrics
3. Add parameter for vertical position of slider
4. Export figure as html
5. convert figure generation into a function

Source: https://plot.ly/python/table/
'''

import plotly.graph_objects as go
import numpy as np
import pandas as pd

# render figure in browser
import plotly.io as pio
pio.renderers.default = "browser"

# Create data

# As you add number of columns, make sure you
# adjust slider vertical position accordingly in
# by setting the parameter `y` in the sliders dict
df = pd.DataFrame(
    {
        'threshold': np.arange(0, 1, 0.01),
        'f1_score': np.random.random(100),
        'accuracy': np.random.random(100),
        'precision': np.random.random(100),
        'recall': np.random.random(100),
        'false_pos_rate': np.random.random(100)
    }
)
df = df.round(3)

#################
# Create figure #
#################

fig = go.Figure()

# Add traces, one for each slider step
# This creates multiple plots and adds them to
# the fig object. The slider below will select
# which plot to display
for i in df.iterrows():

    data = [
        list(i[1].index),  # column names
        list(i[1].values)  # column values
    ]

    fig.add_trace(
        go.Table(
            columnorder=[1, 2],
            columnwidth=[50, 100],
            header=dict(
                values=[['Metrics'], ['Values']],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align=['left', 'center'],
                font_size=22,
                height=50
            ),
            cells=dict(
                values=data,
                line_color='darkslategray',
                fill=dict(color=['lightgrey', 'white']),
                align=['left', 'center'],
                font_size=22,
                height=40
            )
        )
    )

# Create the slider here

# Doc for steps:
# https://plot.ly/python/reference/#layout-sliders-items-slider-steps
steps = []
for i in range(len(fig.data)):

    # Set value along slider axis to the value of the
    # threshold chosen in the slider
    thresh_val = fig.data[i].cells.values[1][0]

    step = dict(
        method="restyle",
        args=["visible", [False] * len(fig.data)],
        label=thresh_val
    )

    step["args"][1][i] = True  # Toggle i'th trace to "visible"

    steps.append(step)

# Doc for sliders:
# https://plot.ly/python/reference/#layout-sliders
sliders = [
    dict(
        active=10,
        currentvalue={
            "prefix": "Set threshold: ",
            "font": {"size": 18, "color": "darkblue"}
        },
        pad={"t": 0, "b": 0},
        x=0,            # horizontal position of slider
        y=.60,           # vertical position of slider
        steps=steps,
        len=.4          # length of slider
    )
]

fig.update_layout(
    sliders=sliders
)

# Set this if you want to fix the height and
# width of the figure in the browser. It will
# disable dynamic adjustment of height and width

# fig.update_layout(width=800, height=800)

fig.show()
