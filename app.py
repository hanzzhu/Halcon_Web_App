import base64
import datetime
import time

import plotly
import plotly.figure_factory as ff
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import json
import run_CL as run
import run_OD as run_OD

# Initialise empty lists that will be used to carry values for multiple usage. e.g. plotting graph, passing metrics etc.
iterationList = []
lossList = []
epochOfLossList = []
epochOfTop1ErrorList = []
epochOfMeanAPList = []
TrainSet_top1_error_valueList = []
ValidationSet_top1_error_valueList = []
TrainSet_mean_ap_valueList = []
ValidationSet_mean_ap_valueList = []
metricList = []

# Default stylesheet from official guide. May write own stylesheet with css.
# The stylesheet may be the key for other dash community components to work. e.g. bootstrap
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Initialise the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Icon on top segment of the web.
# Icon image stored within the direct folder for ease of transfer.
image_filename = 'icon.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# Project names that to be shown
CLProjectNames = ','.join(run.CLProjectList)
ODProjectNames = ','.join(run_OD.ODProjectList)

# Initialise web layout.
app.layout = html.Div(
    [
        html.Center(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), width='80', height='70')),
        # Title
        html.H1('CHaDLE ',
                style={
                    "font": 'verdana',
                    'textAlign': 'center',
                    'color': 'Black'
                }
                ),

        # Tabs for CL and OD
        html.Div(
            [
                dcc.Tabs(
                    id='AllTab',
                    value='AllTab',
                    children=
                    [

                        # Classification Tab

                        dcc.Tab(
                            label='Classification',
                            value='CLTab',
                            children=
                            [
                                html.Div([
                                    html.Div(
                                        [
                                            # Main control panel - project name, device, pretrained model, buttons.
                                            html.Div(
                                                [
                                                    html.Th(
                                                        children='Available Classification Projects: ' + CLProjectNames,
                                                        colSpan="1"),
                                                    html.Br(),

                                                    "Project Name:",

                                                    dcc.Input(
                                                        id='ProjectName_CL', value='Animals', type='text'
                                                    ),
                                                    html.Br(),
                                                    "Training Device:",
                                                    dcc.RadioItems(
                                                        id='Runtime_CL',
                                                        options=[{'label': i, 'value': i} for i in ['cpu', 'gpu']],
                                                        value='cpu',
                                                        labelStyle={'display': 'inline-block'}
                                                    ),

                                                    "Pretrained Model:",
                                                    dcc.Dropdown(
                                                        id='PretrainedModel_CL',
                                                        options=[{'label': i, 'value': i} for i in
                                                                 ["classifier_enhanced", "classifier_compact"]],
                                                        value='classifier_compact'
                                                    ),

                                                    # Empty space wrapped in Div. Single Br() seems not working sometimes.
                                                    # There should be better ways to add these empty lines/ arrange the layout.
                                                    html.Div([html.Br()]),
                                                    html.Div([html.Br()]),

                                                    # Button for training
                                                    html.Button(
                                                        id='operation_button_CL',
                                                        n_clicks=0,
                                                        children='Start Training'

                                                    ),
                                                    html.Div([html.Br()]),
                                                    html.Div([html.Br()]),
                                                    html.Div([html.Br()]),
                                                    dcc.Loading(
                                                        id="loading-1",
                                                        type="default",
                                                        children=[html.Div(id="Training_loading_CL"),
                                                                  html.Div(id="Evaluation_loading_CL")]
                                                    ),
                                                ],
                                                style={'width': '35%', 'float': 'left', 'display': 'inline-block'}
                                            ),

                                            # Augmentation parameters panel
                                            html.Div(
                                                [
                                                    # Percentage of augmentation. Mandatory. 0% means not activated
                                                    # To improve, use dcc.input instead of Label for showing percentage.
                                                    # However complicated.
                                                    html.Div(
                                                        [
                                                            html.Label(id='AugmentationPercentage_CL',
                                                                       style={'color': 'black', 'padding': '6px',
                                                                              'font-family': 'Arial'}
                                                                       )
                                                        ],
                                                        style={'display': 'inline'}
                                                    ),
                                                    html.Div(
                                                        dcc.Slider(
                                                            id='AugmentationPercentage_CL_slider',
                                                            min=0,
                                                            max=100,
                                                            value=0,
                                                            marks={
                                                                0: {'label': '0%', },
                                                                25: {'label': '25%'},
                                                                50: {'label': '50%', 'style': {'color': '#77b0b1'}},
                                                                75: {'label': '75%'},
                                                                100: {'label': '100%'}
                                                            },
                                                        ),
                                                        style={'width': '100%', 'float': 'left'},
                                                    ),
                                                    html.Div([html.Br()]),
                                                    html.Div([html.Br()]),
                                                    # Rotation & Rotation Range checkbox and input. If not activated, placeholder
                                                    # will show 'disabled', same as following parameters.
                                                    dcc.Checklist(
                                                        id='Rotation_CL_Switch',
                                                        options=[
                                                            {
                                                                'label': 'Activate Rotation', 'value': 0
                                                            }
                                                        ],
                                                        value=[],
                                                        style={'color': '#C04000'}
                                                    ),

                                                    html.Label('Rotation(Step of 90 degree)'),
                                                    dcc.Input(
                                                        id='Rotation_CL',
                                                        value='',
                                                        type='number',
                                                        min=-180, max=180,
                                                        step=90,
                                                        disabled=True,
                                                        placeholder="Disabled"
                                                    ),

                                                    html.Label('Rotation Range(Step of 1)'),
                                                    dcc.Input(
                                                        id='RotationRange_CL',
                                                        value='',
                                                        type='number',
                                                        min=0,
                                                        step=1,
                                                        disabled=True,
                                                        placeholder="Disabled"
                                                    ),
                                                    html.Div([html.Br()]),

                                                    # Mirror checkbox and input
                                                    dcc.Checklist(
                                                        id='mirror_CL_Switch',
                                                        options=[
                                                            {
                                                                'label': 'Activate Mirror', 'value': 0
                                                            }
                                                        ],
                                                        value=[],
                                                        style={'color': '#C04000'}
                                                    ),

                                                    html.Label('Mirror (off,c,r,rc)'),
                                                    dcc.Input(
                                                        id='mirror_CL',
                                                        value='',
                                                        type='text',
                                                        disabled=True,
                                                        placeholder="Disabled"
                                                    ),
                                                    html.Div([html.Br()]),

                                                    # Brightness Variation and Variation Spot checkbox and input
                                                    dcc.Checklist(
                                                        id='BrightnessVariation_CL_Switch',
                                                        options=[
                                                            {
                                                                'label': 'Activate Brightness Variation', 'value': 0
                                                            }
                                                        ],
                                                        value=[],
                                                        style={'color': '#C04000'}
                                                    ),

                                                    html.Label('Brightness Variation'),
                                                    dcc.Input(
                                                        id='BrightnessVariation_CL',
                                                        value='',
                                                        type='number',
                                                        min=-100,
                                                        max=100,
                                                        step=1,
                                                        disabled=True,
                                                        placeholder="Disabled"
                                                    ),

                                                    html.Label('Brightness Variation Spot'),
                                                    dcc.Input(
                                                        id='BrightnessVariationSpot_CL',
                                                        value='',
                                                        type='number',
                                                        min=0,
                                                        step=1,
                                                        disabled=True,
                                                        placeholder="Disabled"
                                                    ),
                                                    html.Div([html.Br()]),

                                                    # Crop Percentage and Crop Pixel checkbox and input
                                                    dcc.Checklist(
                                                        id='Crop_CL_Switch',
                                                        options=[
                                                            {
                                                                'label': 'Activate Crop', 'value': 0
                                                            }
                                                        ],
                                                        value=[],
                                                        style={'color': '#C04000'}
                                                    ),

                                                    html.Label('Crop Percentage'),
                                                    dcc.Input(
                                                        id='CropPercentage_CL',
                                                        value='',
                                                        type='number',
                                                        min=0,
                                                        max=100,
                                                        step=1,
                                                        disabled=True,
                                                        placeholder="Disabled"
                                                    ),

                                                    html.Label('Crop Pixel'),
                                                    dcc.Input(
                                                        id='CropPixel_CL',
                                                        value='',
                                                        type='number',
                                                        min=0,
                                                        step=1,
                                                        disabled=True,
                                                        placeholder="Disabled"
                                                    ),
                                                    html.Div([html.Br()]),
                                                    # Ignore Direction checkbox and input

                                                    dcc.Checklist(
                                                        id='Direction_CL_Switch',
                                                        options=[
                                                            {
                                                                'label': 'Ignore Direction', 'value': 0
                                                            }
                                                        ],
                                                        value=[],
                                                        style={'color': '#C04000'}
                                                    ),
                                                    html.Div([html.Br()]),
                                                    dcc.Checklist(
                                                        id='ClassID_CL_Switch',
                                                        options=[
                                                            {
                                                                'label': 'Class IDs No Orientation Exist', 'value': 0
                                                            }
                                                        ],
                                                        value=[],
                                                        style={'color': '#C04000'}
                                                    ),
                                                    html.Label('Class IDs No Orientation'),
                                                    dcc.Input(
                                                        id='ClassIDsNoOrientation',
                                                        value='',
                                                        type='text',
                                                        disabled=True,
                                                        placeholder="Disabled"
                                                    ),
                                                ],
                                                style={'width': '35%', 'display': 'inline-block'}
                                            ),

                                            # Image and training detail parameters panel
                                            # Settings have clearer format, so no extra indent.
                                            html.Div(
                                                [
                                                    html.Label('Image Width'),
                                                    dcc.Input(id='ImWidth_CL', value='100 ', type='number', min=0,
                                                              step=1, ),
                                                    html.Label('Image Height'),
                                                    dcc.Input(id='ImHeight_CL', value='100', type='number', min=0,
                                                              step=1, ),
                                                    html.Label('Image Channel'),
                                                    dcc.Input(id='ImChannel_CL', value='3', type='number', min=0,
                                                              step=1, ),
                                                    html.Label('Batch Size'),
                                                    dcc.Input(id='BatchSize_CL', value='1', type='number', min=0,
                                                              step=1, ),
                                                    html.Label('Initial Learning Rate'),
                                                    dcc.Input(id='InitialLearningRate_CL', value='0.001', type='number',
                                                              min=0,
                                                              step=0.00001, ),
                                                    html.Label('Momentum'),
                                                    dcc.Input(id='Momentum_CL', value='0.09', type='number', min=0,
                                                              step=0.00001, ),
                                                    html.Label('Number of Epochs'),
                                                    dcc.Input(id='NumEpochs_CL', value='2', type='number', min=0,
                                                              step=1, ),
                                                    html.Label('Change Learning Rate @ Epochs'),
                                                    dcc.Input(id='ChangeLearningRateEpochs_CL', value='50,100',
                                                              type='text'),
                                                    html.Label('Learning Rate Schedule'),
                                                    dcc.Input(id='lr_change_CL', value='0.01,0.05', type='text'),
                                                    html.Label('Regularisation Constant'),
                                                    dcc.Input(id='WeightPrior_CL', value='0.001', type='number', min=0,
                                                              step=0.00001, ),
                                                    html.Label('Class Penalty'),
                                                    dcc.Input(id='class_penalty_CL', value='0,0', type='text'),
                                                ],
                                                style={'width': '30%', 'display': 'inline-block'}
                                            ),
                                            html.Div([html.Br()]),
                                            html.Div([html.Br()]),
                                        ],
                                        style={'width': '45%', 'display': 'inline-block', 'float': 'left'}
                                    ),

                                    # Graph Plotter
                                    html.Div(
                                        [
                                            html.H4('Training Monitor',
                                                    style={
                                                        "font": 'Helvetica',
                                                        'textAlign': 'center',
                                                        'color': 'Black'
                                                    }
                                                    ),
                                            html.Div(id='metrics_CL',
                                                     style={
                                                         "font": 'Helvetica',
                                                         'textAlign': 'center',
                                                         'color': 'Blue'
                                                     }
                                                     ),
                                            dcc.Graph(id='iteration_loss_graph_CL'),
                                            dcc.Graph(id='top1_error_graph_CL'),

                                            # Interval for this section will be sent to respective call-back for running.
                                            # E.g. in this case, the monitoring graphs will be updated every 1000 milisec.
                                            dcc.Interval(
                                                id='interval_graph_CL',
                                                interval=1 * 1000,  # in milliseconds
                                                n_intervals=0,
                                                disabled=False,
                                            )
                                        ],
                                        style={'width': '55%', 'float': 'right'}
                                    ),
                                ]),
                                html.Div([html.Br()]),
                                html.Div([html.Br()]),
                                html.Div([html.Br()]),
                                html.Div([html.Br()]),

                                # Evaluation
                                # Pending to use or not.
                                html.Div([
                                    html.Div(
                                        [
                                            html.H4('Evaluation Graph', style={'float': 'center'}),
                                            html.Button(
                                                id='evaluation_button_CL',
                                                n_clicks=0,
                                                children='Evaluation'
                                            ),
                                        ],
                                        style={'width': '75%', 'float': 'right'}
                                    ),

                                    html.Div([html.Br()]),
                                    html.Div([html.Br()]),

                                    html.Div(
                                        [
                                            html.Div(id='evaluation_text_CL'),
                                            dcc.Graph(id='evaluation_graph_CL'),
                                        ],
                                        style={'width': '60%', 'float': 'left', }
                                    ),

                                    dcc.Interval(
                                        id='interval-evaluation_CL',
                                        interval=1 * 1000,  # in milliseconds
                                        n_intervals=0
                                    )
                                ]),

                                html.Div(id='Operation_output_CL'),
                                html.Div(id='makeJson_CL'),
                            ],
                            style={'display': 'inline-block'}
                        ),

                        # Object Detection Tab

                        dcc.Tab(
                            label='Object Detection',
                            value='ODTab',
                            children=
                            [
                                # Basic inputs
                                html.Div(
                                    [
                                        html.Th(children='Available Object Detection Projects: ' + ODProjectNames,
                                                colSpan="1"),
                                        html.Br(),

                                        "Project Name:",

                                        dcc.Input(
                                            id='ProjectName_OD', value='NTBW', type='text'
                                        ),

                                        html.Br(),

                                        "Training Device:",
                                        dcc.RadioItems(
                                            id='Runtime_OD',
                                            options=[{'label': i, 'value': i} for i in ['cpu', 'gpu']],
                                            value='cpu',
                                            labelStyle={'display': 'inline-block'}
                                        ),

                                        "Pretrained Model:",
                                        dcc.Dropdown(
                                            id='PretrainedModel_OD',
                                            options=[{'label': i, 'value': i} for i in
                                                     ["classifier_enhanced", "classifier_compact"]],
                                            value='classifier_compact'
                                        ),
                                    ],
                                    style={'width': '15%', 'display': 'inline-block'}
                                ),

                                html.Br(),
                                html.Br(),

                                # Parameters inputs
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label('Number of Classes'),
                                                dcc.Input(id='NumClasses_OD', value='5', type='number', min=0,
                                                          step=1),
                                                html.Label('Image Width'),
                                                dcc.Input(id='ImWidth_OD', value='960', type='number', min=0, step=1, ),
                                                html.Label('Image Height'),
                                                dcc.Input(id='ImHeight_OD', value='1024', type='number', min=0,
                                                          step=1),
                                                html.Label('Image Channel'),
                                                dcc.Input(id='ImChannel_OD', value='3', type='number', min=0, step=1),

                                                html.Label('Capacity'),
                                                dcc.Input(id='Capacity_OD', value='medium', type='text', min=0,
                                                          step=1),
                                                html.Label('Instance Type'),
                                                dcc.Input(id='InstanceType_OD', value='rectangle1', type='text', min=0,
                                                          step=1),
                                                html.Label('Training Percent'),
                                                dcc.Input(id='TrainingPercent_OD', value='75', type='number', min=0,
                                                          step=1),
                                                html.Label('Validation Percent'),
                                                dcc.Input(id='ValidationPercent_OD', value='15', type='number', min=0,
                                                          step=1)
                                            ],
                                            style={'width': '15%', 'display': 'inline-block'}),

                                        html.Div(
                                            [
                                                html.Label('Batch Size'),
                                                dcc.Input(id='BatchSize_OD', value='10', type='number', min=0,
                                                          step=1, ),
                                                html.Label('Initial Learning Rate'),
                                                dcc.Input(id='InitialLearningRate_OD', value='0.001', type='number',
                                                          min=0,
                                                          step=0.00001, ),
                                                html.Label('Momentum'),
                                                dcc.Input(id='Momentum_OD', value='0.09', type='number', min=0,
                                                          step=0.00001, ),
                                                html.Label('Number of Epochs'),
                                                dcc.Input(id='NumEpochs_OD', value='2', type='number', min=0, step=1, ),
                                                html.Label('Change Learning Rate @ Epochs'),
                                                dcc.Input(id='ChangeLearningRateEpochs_OD', value='50,100',
                                                          type='text'),
                                                html.Label('Learning Rate Schedule'),
                                                dcc.Input(id='lr_change_OD', value='0.01,0.05', type='text'),
                                                html.Label('Regularisation Constant'),
                                                dcc.Input(id='WeightPrior_OD', value='0.001', type='number', min=0,
                                                          step=0.00001, ),
                                                html.Label('Class Penalty'),
                                                dcc.Input(id='class_penalty_OD', value='0,0', type='text')
                                            ],
                                            style={'width': '15%', 'display': 'inline-block'}),

                                        html.Div(
                                            [
                                                html.Label('Augmentation Percentage'),
                                                dcc.Input(id='AugmentationPercentage_OD', value='100', type='number',
                                                          min=0,
                                                          max=100,
                                                          step=1),
                                                html.Label('Rotation'),
                                                dcc.Input(id='Rotation_OD', value='90', type='number', min=-180,
                                                          max=180,
                                                          step=90),
                                                html.Label('Mirror (off,c,r,rc)'),
                                                dcc.Input(id='mirror_OD', value='off', type='text'),
                                                html.Label('Brightness Variation'),
                                                dcc.Input(id='BrightnessVariation_OD', value='0', type='number',
                                                          min=-100,
                                                          max=100,
                                                          step=1),
                                                html.Label('Brightness Variation Spot'),
                                                dcc.Input(id='BrightnessVariationSpot_OD', value='0', type='number',
                                                          min=-100,
                                                          max=100,
                                                          step=1),
                                                html.Label('Rotation Range (Step of 1)'),
                                                dcc.Input(id='RotationRange_OD', value='10', type='number', min=1,
                                                          step=1),
                                                html.Label('Bbox heads weight'),
                                                dcc.Input(id='BboxHeads_OD', value='', type='number', min=1, step=1),
                                                html.Label('Class heads weight'),
                                                dcc.Input(id='ClassHeads_OD', value='', type='number', min=1, step=1)
                                            ],
                                            style={'width': '15%', 'float': 'initial', 'display': 'inline-block',
                                                   }
                                        ),

                                        # Estimated Value show and input
                                        html.Div(
                                            [
                                                html.H4('Halcon estimated values'),
                                                html.P('Key in new desired value or leave it empty: '),
                                                html.Br(),
                                                html.Div([html.P('Min Level: '),

                                                          html.Div([html.Div(id='MinLevel_OD'), ],
                                                                   style={"font": 'Helvetica', 'color': 'Blue'}),
                                                          dcc.Input(id='MinLevel_Input_OD', placeholder='Integer',
                                                                    type='number',
                                                                    min=0,
                                                                    step=1,
                                                                    debounce=True), ]),
                                                html.Br(),
                                                html.Div([html.P('Max Level: '),

                                                          html.Div([html.Div(id='MaxLevel_OD'), ],
                                                                   style={"font": 'Helvetica', 'color': 'Blue'}),
                                                          dcc.Input(id='MaxLevel_Input_OD', placeholder='Integer',
                                                                    type='number',
                                                                    min=0,
                                                                    step=1,
                                                                    debounce=True), ]),
                                                html.Br(),
                                                html.Div([html.P('Anchor Number of Subscales: '),
                                                          html.Div([html.Div(id='AnchorNumSubscales_OD'), ],
                                                                   style={"font": 'Helvetica', 'color': 'Blue'}),
                                                          dcc.Input(id='AnchorNumSubscales_Input_OD',
                                                                    placeholder='Integer',
                                                                    type='number',
                                                                    min=0,
                                                                    step=1,
                                                                    debounce=True), ]),
                                                html.Br(),
                                                html.Div([html.P('Anchor Aspect Ratios (min,max,mean,deviation): '),
                                                          html.Div([html.Div(id='AnchorAspectRatios_OD'), ],
                                                                   style={"font": 'Helvetica', 'color': 'Blue'}),
                                                          dcc.Input(id='AnchorAspectRatios_Input_OD',
                                                                    placeholder='List (0.720, 1.475, 2.125, 2.753)',
                                                                    type='text', min=0, debounce=True,
                                                                    style={'width': '50%', }), ]),

                                                # if user wanna change, type in the desired value.

                                                # value = Best value among 4 read by halcon
                                                # label the value,
                                            ],
                                            style={'width': '40%', 'float': 'right'},
                                        ),
                                    ]
                                ),

                                html.Br(),
                                html.Br(),
                                html.Br(),

                                dcc.Loading(
                                    id="loading_OD",
                                    type="default",
                                    children=[html.Div(id="Training_loading_OD"),
                                              html.Div(id="Estimate_values_loading_OD")]
                                ),
                                html.Br(),

                                # Buttons
                                html.Div(
                                    [
                                        html.Button(id='estimate_button_OD', n_clicks=0,
                                                    children='Halcon Estimate Values'),
                                        html.Button(id='operation_button_OD', n_clicks=0, children='Train'),
                                        html.Button(id='evaluation_button_OD', n_clicks=0, children='Evaluation'),
                                    ],
                                    style={'display': 'flex',
                                           'justify-content': 'center',
                                           'align-items': 'center',
                                           'height': '100px',
                                           }
                                ),
                                # Dummy output segment reserved for training results.
                                html.Div(
                                    [
                                        html.Label(id='training_output_OD'),
                                    ],
                                    style={'display': 'flex',
                                           'justify-content': 'center',
                                           'align-items': 'center',
                                           'height': '50px'
                                           },
                                ),

                                # Evaluation Graph
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H2('Evaluation Graph Coming Soon...',
                                                        style={
                                                            "font": 'Helvetica',
                                                            'textAlign': 'center',
                                                            'color': 'Black'
                                                        }
                                                        ),
                                                html.Div(id='evaluation_text_OD'),
                                                dcc.Graph(id='evaluation_graph_OD'),
                                            ],
                                            style={'width': '100%', 'float': 'initial'}
                                        ),
                                        dcc.Interval(
                                            id='interval-evaluation_OD',
                                            interval=1 * 1000,  # in milliseconds
                                            n_intervals=0
                                        )
                                    ],
                                ),

                                # OD training monitor graph plotter
                                html.Div(
                                    [
                                        html.H1('CHaDLE Training Monitor - Object Detection',
                                                style={
                                                    "font": 'Helvetica',
                                                    'textAlign': 'center',
                                                    'color': 'Black'
                                                }),
                                        html.Div(id='metrics_OD',
                                                 style={
                                                     "font": 'Helvetica',
                                                     'textAlign': 'center',
                                                     'color': 'Blue'
                                                 }),
                                        dcc.Graph(id='iteration_loss_graph_OD'),
                                        dcc.Graph(id='mean_ap_graph_OD'),
                                        dcc.Interval(
                                            id='interval_graph_OD',
                                            interval=1 * 1000,  # in milliseconds
                                            n_intervals=0
                                        )
                                    ]
                                )
                            ],
                        ),
                    ]
                ),
            ]
        ),
    ]
)


############################################################################################################
############################################## Call Backs ##################################################
############################################################################################################


############################################################################################################
############################################## Switches ####################################################
############################################################################################################

# Augmentation slider
@app.callback(
    Output('AugmentationPercentage_CL', 'children'),
    Input('AugmentationPercentage_CL_slider', 'value'),
)
def update_output(value):
    return 'Augmentation Percentage: {}%'.format(value)


# Rotation switch
@app.callback(
    Output("Rotation_CL", "disabled"),
    Output("Rotation_CL", "placeholder"),
    Output("RotationRange_CL", "disabled"),
    Output("RotationRange_CL", "placeholder"),
    Input("Rotation_CL_Switch", "value"),
    State("Rotation_CL", "disabled"),
    State("Rotation_CL", "placeholder"),
    State("RotationRange_CL", "disabled"),
    State("RotationRange_CL", "placeholder")
)
def Rotation_CL_switch(Rotation_CL_Switch, Rotation_CL_disabled, Rotation_CL_placeholder, RotationRange_CL_disabled,
                       RotationRange_CL_placeholder):
    # The switch controls both Rotation and Rotation Range inputs
    if not Rotation_CL_Switch:
        Rotation_CL_disabled = True
        RotationRange_CL_disabled = True
        Rotation_CL_placeholder = 'Disabled'
        RotationRange_CL_placeholder = 'Disabled'

    elif Rotation_CL_Switch == [0]:
        Rotation_CL_disabled = False
        RotationRange_CL_disabled = False
        Rotation_CL_placeholder = ''
        RotationRange_CL_placeholder = ''

    # Note that must return in sequence according to Output()
    return Rotation_CL_disabled, Rotation_CL_placeholder, RotationRange_CL_disabled, RotationRange_CL_placeholder


# Mirror switch
@app.callback(
    Output("mirror_CL", "disabled"),
    Output("mirror_CL", "placeholder"),
    Input("mirror_CL_Switch", "value"),
    State("mirror_CL", "disabled"),
    State("mirror_CL", "placeholder")
)
def mirror_CL_switch(mirror_CL_Switch, disabled, placeholder):
    if not mirror_CL_Switch:
        disabled = True
        placeholder = 'Disabled'
    elif mirror_CL_Switch == [0]:
        disabled = False
        placeholder = ''

    return disabled, placeholder


# Brightness switch
@app.callback(
    Output("BrightnessVariation_CL", "disabled"),
    Output("BrightnessVariation_CL", "placeholder"),
    Output("BrightnessVariationSpot_CL", "disabled"),
    Output("BrightnessVariationSpot_CL", "placeholder"),
    Input("BrightnessVariation_CL_Switch", "value"),
    State("BrightnessVariation_CL", "disabled"),
    State("BrightnessVariation_CL", "placeholder"),
    State("BrightnessVariationSpot_CL", "disabled"),
    State("BrightnessVariationSpot_CL", "placeholder")
)
def Brightness_Variation_CL_switch(BrightnessVariation_CL_Switch, BrightnessVariation_CL_disabled,
                                   BrightnessVariation_CL_placeholder, BrightnessVariationSpot_CL_disabled,
                                   BrightnessVariationSpot_CL_placeholder):
    # The switch controls both inputs
    if not BrightnessVariation_CL_Switch:
        BrightnessVariation_CL_disabled = True
        BrightnessVariationSpot_CL_disabled = True
        BrightnessVariation_CL_placeholder = 'Disabled'
        BrightnessVariationSpot_CL_placeholder = 'Disabled'

    elif BrightnessVariation_CL_Switch == [0]:
        BrightnessVariation_CL_disabled = False
        BrightnessVariationSpot_CL_disabled = False
        BrightnessVariation_CL_placeholder = ''
        BrightnessVariationSpot_CL_placeholder = ''

    return BrightnessVariation_CL_disabled, BrightnessVariation_CL_placeholder, BrightnessVariationSpot_CL_disabled, BrightnessVariationSpot_CL_placeholder


# Crop switch
@app.callback(
    Output("CropPercentage_CL", "disabled"),
    Output("CropPercentage_CL", "placeholder"),
    Output("CropPixel_CL", "disabled"),
    Output("CropPixel_CL", "placeholder"),
    Input("Crop_CL_Switch", "value"),
    State("CropPercentage_CL", "disabled"),
    State("CropPercentage_CL", "placeholder"),
    State("CropPixel_CL", "disabled"),
    State("CropPixel_CL", "placeholder")
)
def Crop_CL_switch(Crop_CL_Switch, CropPercentage_CL_disabled, CropPercentage_CL_placeholder, CropPixel_CL_disabled,
                   CropPixel_CL_placeholder):
    if not Crop_CL_Switch:
        CropPercentage_CL_disabled = True
        CropPixel_CL_disabled = True
        CropPercentage_CL_placeholder = 'Disabled'
        CropPixel_CL_placeholder = 'Disabled'

    elif Crop_CL_Switch == [0]:
        CropPercentage_CL_disabled = False
        CropPixel_CL_disabled = False
        CropPercentage_CL_placeholder = ''
        CropPixel_CL_placeholder = ''

    return CropPercentage_CL_disabled, CropPercentage_CL_placeholder, CropPixel_CL_disabled, CropPixel_CL_placeholder


# Class ID switch
@app.callback(
    Output("ClassIDsNoOrientation", "disabled"),
    Output("ClassIDsNoOrientation", "placeholder"),
    Input("ClassID_CL_Switch", "value"),
    State("ClassIDsNoOrientation", "disabled"),
    State("ClassIDsNoOrientation", "placeholder")
)
def ClassIDs_CL_switch(ClassID_CL_Switch, disabled, placeholder):
    if not ClassID_CL_Switch:
        disabled = True
        placeholder = 'Disabled'
    elif ClassID_CL_Switch == [0]:
        disabled = False
        placeholder = ''

    return disabled, placeholder


############################################################################################################
############################################ Classification ################################################
############################################################################################################

# Main CL call back that combines augmentation,pre-process and training
@app.callback(Output('Operation_output_CL', 'children'),
              Output("Training_loading_CL", "children"),
              Input('operation_button_CL', 'n_clicks'),
              State('Rotation_CL_Switch', 'value'),
              State('mirror_CL_Switch', 'value'),
              State('BrightnessVariation_CL_Switch', 'value'),
              State('Crop_CL_Switch', 'value'),
              State('Direction_CL_Switch', 'value'),
              State('ProjectName_CL', 'value'),
              State('Runtime_CL', 'value'),
              State('PretrainedModel_CL', 'value'),
              State('ImWidth_CL', 'value'),
              State('ImHeight_CL', 'value'),
              State('ImChannel_CL', 'value'),
              State('BatchSize_CL', 'value'),
              State('InitialLearningRate_CL', 'value'),
              State('Momentum_CL', 'value'),
              State('NumEpochs_CL', 'value'),
              State('ChangeLearningRateEpochs_CL', 'value'),
              State('lr_change_CL', 'value'),
              State('WeightPrior_CL', 'value'),
              State('class_penalty_CL', 'value'),
              State('AugmentationPercentage_CL_slider', 'value'),
              State('Rotation_CL', 'value'),
              State('mirror_CL', 'value'),
              State('BrightnessVariation_CL', 'value'),
              State('BrightnessVariationSpot_CL', 'value'),
              State('RotationRange_CL', 'value'),
              State('CropPercentage_CL', 'value'),
              State('CropPixel_CL', 'value'),
              )
def operation_CL(operation_button_CL, Rotation_CL_Switch, mirror_CL_Switch, BrightnessVariation_CL_Switch,
                 Crop_CL_Switch, Direction_CL_Switch, ProjectName_CL, Runtime_CL, PretrainedModel_CL, ImWidth_CL,
                 ImHeight_CL, ImChannel_CL, BatchSize_CL, InitialLearningRate_CL, Momentum_CL, NumEpochs_CL,
                 ChangeLearningRateEpochs_CL, lr_change_CL, WeightPrior_CL,
                 class_penalty_CL, AugmentationPercentage_CL, Rotation_CL, mirror_CL, BrightnessVariation_CL,
                 BrightnessVariationSpot_CL, RotationRange_CL, CropPercentage_CL, CropPixel_CL):
    # Defines the button trigger context
    ctx_operation_CL = dash.callback_context
    if not ctx_operation_CL.triggered:
        button_id = 'Null'
    else:
        button_id = ctx_operation_CL.triggered[0]['prop_id'].split('.')[0]

    # Check which button clicked
    if button_id == 'Null':
        raise PreventUpdate
    else:
        if button_id == 'operation_button_CL':

            # Get the chosen project name, and it's directory path
            var = list((x for x in list(map(str.upper, run.CLProjectList)) if ProjectName_CL.upper() in x))
            ProjectDir_CL = run.Chadle_DataDir + '/Classification/' + var[0]

            # Remove the status file from previous training
            # if os.path.exists(ProjectDir_CL + '/status.txt'):
            #    os.remove(ProjectDir_CL + '/status.txt')

            # Record all the parameter input
            ParameterDict = {'ProjectName': ProjectName_CL,
                             'Runtime': Runtime_CL, 'PretrainedModel': PretrainedModel_CL, 'ImWidth': ImWidth_CL,
                             'ImHeight': ImHeight_CL,
                             'ImChannel': ImChannel_CL,
                             'BatchSize': BatchSize_CL, 'InitialLearningRate': InitialLearningRate_CL,
                             'Momentum': Momentum_CL,
                             'NumEpochs': NumEpochs_CL,
                             'ChangeLearningRateEpochs': ChangeLearningRateEpochs_CL, 'lr_change': lr_change_CL,
                             'WeightPrior': WeightPrior_CL,
                             'class_penalty': class_penalty_CL, 'AugmentationPercentage': AugmentationPercentage_CL,
                             'Rotation': Rotation_CL, 'mirror': mirror_CL,
                             'BrightnessVariation': BrightnessVariation_CL,
                             'BrightnessVariationSpot': BrightnessVariationSpot_CL,
                             'RotationRange': RotationRange_CL, 'Crop Percentage': CropPercentage_CL,
                             'Crop Pixel': CropPixel_CL, }

            #  Output as txt file as requested by IT team.
            with open(ProjectDir_CL + '/hyper_parameters.txt', 'w') as outfile:
                json.dump(ParameterDict, outfile)

            # Conduct pre-process, outputs will be in Hdict format storing in lists.
            pre_process_param = run.pre_process_CL(Rotation_CL_Switch, mirror_CL_Switch,
                                                   BrightnessVariation_CL_Switch,
                                                   Crop_CL_Switch, Direction_CL_Switch, ProjectName_CL, Runtime_CL,
                                                   PretrainedModel_CL,
                                                   ImWidth_CL,
                                                   ImHeight_CL,
                                                   ImChannel_CL,
                                                   BatchSize_CL, InitialLearningRate_CL, Momentum_CL, NumEpochs_CL,
                                                   ChangeLearningRateEpochs_CL, lr_change_CL, WeightPrior_CL,
                                                   class_penalty_CL, AugmentationPercentage_CL, Rotation_CL, mirror_CL,
                                                   BrightnessVariation_CL, BrightnessVariationSpot_CL,
                                                   RotationRange_CL, CropPercentage_CL, CropPixel_CL)

            # Get the outputs from pre-process
            DLModelHandle = pre_process_param[0][0]
            DLDataset = pre_process_param[1][0]
            TrainParam = pre_process_param[2][0]
            ProjectDir = pre_process_param[3]

            # Use the outputs for training
            run.training_CL(DLModelHandle, DLDataset, TrainParam,ProjectDir)

            # Use the outputs for metric showing
            metricList.append(DLModelHandle)
            metricList.append(DLDataset)
            metricList.append(TrainParam)

        # Dummy branch reserved for future development
        else:
            i = 1

    return '', ''


# Plot evaluation graph
@app.callback(Output('evaluation_graph_CL', 'figure'),
              Output('Evaluation_loading_CL', 'children'),
              Input('evaluation_button_CL', 'n_clicks'),
              State('ProjectName_CL', 'value'),
              State('ImWidth_CL', 'value'),
              State('ImHeight_CL', 'value'),
              # State('RotationRange', 'value'),
              # State('IgnoreDirection', 'value'),

              )
def evaluation_CL(evaluation_button_CL, ProjectName_CL, ImWidth_CL, ImHeight_CL):
    # Initialise labels
    z = [[0, 0], [0, 0]]
    x = ['Confusion Matrix', 'Confusion Matrix']
    y = ['Confusion Matrix', 'Confusion Matrix']

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # Plot the initial figure
    fig = ff.create_annotated_heatmap([[0, 0], [0, 0]], x=x, y=y, annotation_text=z_text, colorscale='Blues')

    # Button listener
    ctx_evaluation_CL = dash.callback_context
    if not ctx_evaluation_CL.triggered:
        button_id = 'Null'
    else:
        button_id = ctx_evaluation_CL.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'evaluation_button_CL':
        # Conduct evaluation
        evaluationList = run.evaluation_CL(ProjectName_CL, ImWidth_CL, ImHeight_CL)

        # Clear axis
        z.clear()
        x.clear()
        y.clear()
        z_text.clear()

        # Store in the obtained results
        confusion_matrix_List = evaluationList[0]
        mean_precision = evaluationList[1][0]
        mean_recall = evaluationList[2][0]
        mean_f_score = evaluationList[3][0]

        # Format the results
        mean_precision = format(mean_precision, '.3f')
        mean_recall = format(mean_recall, '.3f')
        mean_f_score = format(mean_f_score, '.3f')

        # Label by sub folder names.
        categories = run.getImageCategories(ProjectName_CL, 'Classification')[0]
        labels = run.getImageCategories(ProjectName_CL, 'Classification')[1]
        # threading.Thread(target=evaluation).start()

        # Sub in the lists to plot
        length = len(categories)
        sublist = [confusion_matrix_List[i:i + length] for i in range(0, len(confusion_matrix_List), length)]
        for i in sublist:
            z.append(i)
        for i in categories:
            x.append(i)
            y.append(i)

        # change each element of z to type string for annotations
        # z_text = [[str(y) for y in x] for x in z]

        # set up figure
        z_text = [[str(y) for y in x] for x in z]
        fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Blues')

        # change each element of z to type string for annotations
        # add title
        fig.update_layout(
            title_text='Mean Precision: ' + str(mean_precision) + '\n Mean Recall: ' + str(
                mean_recall) + '\n Mean F Score: ' + str(mean_f_score),
        )

        # add custom xaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Ground Truth",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        fig.add_annotation(dict(font=dict(color="black", size=14),
                                x=-0.1,
                                y=0.5,
                                showarrow=False,
                                text="Prediction",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=50, l=200))

        # add colorbar
        fig['data'][0]['showscale'] = True

    return fig, ' '


# Historical method to produce json file of input parameters
@app.callback(Output('makeJson_CL', 'children'),
              # Input('parameters_out_button', 'n_clicks'),
              Input('ProjectName_CL', 'value'),
              State('Runtime_CL', 'value'),
              State('PretrainedModel_CL', 'value'),
              State('ImWidth_CL', 'value'),
              State('ImHeight_CL', 'value'),
              State('ImChannel_CL', 'value'),
              State('BatchSize_CL', 'value'),
              State('InitialLearningRate_CL', 'value'),
              State('Momentum_CL', 'value'),
              State('NumEpochs_CL', 'value'),
              State('ChangeLearningRateEpochs_CL', 'value'),
              State('lr_change_CL', 'value'),
              State('WeightPrior_CL', 'value'),
              State('class_penalty_CL', 'value'),
              State('AugmentationPercentage_CL_slider', 'value'),
              State('Rotation_CL', 'value'),
              State('mirror_CL', 'value'),
              State('BrightnessVariation_CL', 'value'),
              State('BrightnessVariationSpot_CL', 'value'),

              State('RotationRange_CL', 'value'),
              # State('IgnoreDirection', 'value'),
              # State('ClassIDsNoOrientationExist', 'value'),
              # State('ClassIDsNoOrientation', 'value'),
              )
def makeJson_CL(ProjectName_CL, Runtime_CL, PretrainedModel_CL, ImWidth_CL, ImHeight_CL, ImChannel_CL,
                BatchSize_CL, InitialLearningRate_CL, Momentum_CL, NumEpochs_CL,
                ChangeLearningRateEpochs_CL, lr_change_CL, WeightPrior_CL,
                class_penalty_CL, AugmentationPercentage_CL, Rotation_CL, mirror_CL,
                BrightnessVariation_CL, BrightnessVariationSpot_CL,
                RotationRange_CL):
    ParameterDict = {'ProjectName': ProjectName_CL,
                     'Runtime': Runtime_CL, 'PretrainedModel': PretrainedModel_CL, 'ImWidth': ImWidth_CL,
                     'ImHeight': ImHeight_CL,
                     'ImChannel': ImChannel_CL,
                     'BatchSize': BatchSize_CL, 'InitialLearningRate': InitialLearningRate_CL, 'Momentum': Momentum_CL,
                     'NumEpochs': NumEpochs_CL,
                     'ChangeLearningRateEpochs': ChangeLearningRateEpochs_CL, 'lr_change': lr_change_CL,
                     'WeightPrior': WeightPrior_CL,
                     'class_penalty': class_penalty_CL, 'AugmentationPercentage': AugmentationPercentage_CL,
                     'Rotation': Rotation_CL, 'mirror': mirror_CL,
                     'BrightnessVariation': BrightnessVariation_CL,
                     'BrightnessVariationSpot': BrightnessVariationSpot_CL,
                     'RotationRange': RotationRange_CL, }
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'Null'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'parameters_out_button':
        with open('parameters_json.txt', 'w') as outfile:
            json.dump(ParameterDict, outfile)
        return 'To json done!'




# Update metrics in the monitoring graph
@app.callback(Output('metrics_CL', 'children'),
              Input('interval_graph_CL', 'n_intervals'),
              State('ProjectName_CL', 'value'),
              State('NumEpochs_CL', 'value'),
)
def update_metrics_CL(n,ProjectName_CL, NumEpochs_CL):
    var = list((x for x in list(map(str.upper, run.CLProjectList)) if ProjectName_CL.upper() in x))
    ProjectDir_CL = run.Chadle_DataDir + '/Classification/' + var[0]

    # Indication Text configuration
    # Extract data from Hdict and show as texts.
    style = {'padding': '5px', 'fontSize': '16px'}

    # Call the function that does the job
    get_metrics = run.get_TrainInfo_CL()
    if get_metrics:
        time_elapsed = get_metrics[0]
        time_remaining = get_metrics[1]
        epoch_metrics = get_metrics[2]
    # If no metrics to show, will show 0 for formatting
    else:
        time_elapsed = 0
        time_remaining = 0
        epoch_metrics = 0

    # Values to be indicated in status.txt
    PercentageEpoch = (epoch_metrics / int(NumEpochs_CL)) * 100
    MetricList = [
        html.Span('Time Elapsed: {}'.format(str(datetime.timedelta(seconds=int(time_elapsed)))),
                  style=style),
        html.Span('Time Remaining: {}'.format(time_remaining), style=style),
        html.Span('Current Epoch: {}'.format(epoch_metrics), style=style)
    ]

    if PercentageEpoch < 100:
        Status = 'In Progress'
        # Store the values into a dict and produce the txt file with json form.
        StatusDict = [epoch_metrics, str(PercentageEpoch), time_remaining, Status]
        with open(ProjectDir_CL + '/Training Stats.txt', 'w') as outfile:
            json.dump(StatusDict, outfile)

    elif PercentageEpoch == 100:
        Status = 'Done'
        IntervalDisabled = False
        # PercentageEpoch = 101
        # Store the values into a dict and produce the txt file with json form.
        StatusDict = [epoch_metrics, str(100), time_remaining, Status]
        with open(ProjectDir_CL + '/Training Stats.txt', 'w') as outfile:
            json.dump(StatusDict, outfile)

    return MetricList


# Plotting iteration-loss-graph for CL
# Multiple components can update everytime interval gets fired.
@app.callback(Output('iteration_loss_graph_CL', 'figure'),
              Input('interval_graph_CL', 'n_intervals'))
def iteration_loss_graph_CL(n):
    # Loss Graph configuration
    # Using plotly subplots. May consider changing to others.
    iteration_loss_graph_fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=1)
    iteration_loss_graph_fig['layout']['margin'] = {
        'l': 80, 'r': 80, 'b': 50, 't': 80, 'autoexpand': False,
    }
    iteration_loss_graph_fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left', 'title': 'Loss-Iteration Graph'}
    iteration_loss_graph_fig.update_layout(legend_title_text=123)
    iteration_loss_graph_fig.update_xaxes(title_text="Iteration", row=1, col=1)
    iteration_loss_graph_fig.update_yaxes(title_text="Loss", row=1, col=1)

    # If Hdict files does not exist, clear graph and lists for plotting.
    # Therefore, could reset graph by deleting the Hdict files.
    getTrainInfo = run.get_TrainInfo_CL()
    if not getTrainInfo:

        iterationList.clear()
        epochOfLossList.clear()
        lossList.clear()
    else:
        epoch_TrainInfo = getTrainInfo[2]
        loss = getTrainInfo[3]
        iteration = getTrainInfo[4]

        # Avoid duplicate output from Halcon.
        # Interval for this web app is set to 1 sec. However feedback from Halcon may take up tp 5 secs.
        # Using <in> with list, average time complexity: O(n)
        # if iteration not in iterationList:
        epochOfLossList.append(epoch_TrainInfo)
        lossList.append(loss)
        iterationList.append(iteration)

    # Add the values to graph and start plotting.
    iteration_loss_graph_fig.append_trace({

        'x': epochOfLossList,
        'y': lossList,
        'text': iterationList,
        'name': 'iteration vs loss',
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)

    return iteration_loss_graph_fig


# Plotting top1-error-graph for CL
@app.callback(Output('top1_error_graph_CL', 'figure'),
              Input('interval_graph_CL', 'n_intervals'))
def top1_error_graph_CL(n):
    # Top1 Error Graph configuration.
    # Using plotly subplots. May consider changing to others.
    top1_error_graph_fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=1, )
    top1_error_graph_fig['layout']['margin'] = {
        'l': 80, 'r': 80, 'b': 100, 't': 80, 'autoexpand': False,
    }
    top1_error_graph_fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    top1_error_graph_fig.update_xaxes(title_text="Epoch", row=1, col=1)
    top1_error_graph_fig.update_yaxes(title_text="Top1 Error", row=1, col=1)

    # If Hdict files does not exist, clear graph and lists for plotting.
    # Therefore, could reset graph by deleting the Hdict files.
    getEvaluationInfo = run.get_EvaluationInfo_CL()
    if not getEvaluationInfo:

        TrainSet_top1_error_valueList.clear()
        ValidationSet_top1_error_valueList.clear()
        epochOfTop1ErrorList.clear()
    else:
        epoch_EvaluationInfo = getEvaluationInfo[0]
        TrainSet_top1_error_value = getEvaluationInfo[1]
        ValidationSet_top1_error_value = getEvaluationInfo[2]

        # Avoid duplicate output from Halcon.
        # Interval for this web app is set to 1 sec. However feedback from Halcon may take up tp 5 secs.
        # Using <in> with list, average time complexity: O(n)
        if TrainSet_top1_error_value not in TrainSet_top1_error_valueList:
            epochOfTop1ErrorList.append(epoch_EvaluationInfo)
            TrainSet_top1_error_valueList.append(TrainSet_top1_error_value)
            ValidationSet_top1_error_valueList.append(ValidationSet_top1_error_value)

        # Add the values to graph and start plotting.
        # Two plots on the same graph.
        top1_error_graph_fig.append_trace({
            'x': epochOfTop1ErrorList,
            'y': TrainSet_top1_error_valueList,

            'name': 'Train Set Top1_error',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)

        top1_error_graph_fig.append_trace({
            'x': epochOfTop1ErrorList,
            'y': ValidationSet_top1_error_valueList,

            'name': 'Validation Set Top1_error',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)
    return top1_error_graph_fig


############################################################################################################
########################################### Object Detection ###############################################
############################################################################################################
@app.callback(Output('MinLevel_OD', 'children'),
              Output('MaxLevel_OD', 'children'),
              Output('AnchorNumSubscales_OD', 'children'),
              Output('AnchorAspectRatios_OD', 'children'),
              Output('Estimate_values_loading_OD', 'children'),
              Input('estimate_button_OD', 'n_clicks'),
              State('ImWidth_OD', 'value'),
              State('ImHeight_OD', 'value'),
              State('TrainingPercent_OD', 'value'),
              State('ValidationPercent_OD', 'value'),
              )
def estimate_value_OD(estimate_button_OD, ImWidth_OD, ImHeight_OD, TrainingPercent_OD, ValidationPercent_OD, ):
    Label_data_OD = 'D:/930415/Chadle_Projects/Chadle_Data/Object_Detection/NTBW_Image Analytics/NTBW_Initial_2.hdict'
    ctx_estimate_value_OD = dash.callback_context
    if not ctx_estimate_value_OD.triggered:
        button_id = 'Null'
    else:
        button_id = ctx_estimate_value_OD.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'estimate_button_OD':
        estimate_value = run_OD.estimate_values_OD(ImWidth_OD, ImHeight_OD, TrainingPercent_OD,
                                                   ValidationPercent_OD, Label_data_OD)

        DLDataset_preprocess = (estimate_value[0])
        MinLevel_OD = (estimate_value[1])
        MaxLevel_OD = (estimate_value[2])
        AnchorNumSubscales_OD = (estimate_value[3])

        estimate_value = [round(number, 3) for number in estimate_value[4]]
        print(estimate_value)
        AnchorAspectRatios_OD_String = ", ".join(str(number) for number in estimate_value)
        AnchorAspectRatios_OD = AnchorAspectRatios_OD_String

        return MinLevel_OD, MaxLevel_OD, AnchorNumSubscales_OD, AnchorAspectRatios_OD, ' '

    else:
        return ' ', ' ', ' ', ' ', ' '


@app.callback(Output('training_output_OD', 'children'),

              Output('Training_loading_OD', 'children'),
              Input('operation_button_OD', 'n_clicks'),

              # State('ProjectName_OD', 'value'),
              State('ImWidth_OD', 'value'),
              State('ImHeight_OD', 'value'),
              State('TrainingPercent_OD', 'value'),
              State('ValidationPercent_OD', 'value'),
              State('MinLevel_Input_OD', 'value'),
              State('MaxLevel_Input_OD', 'value'),
              State('AnchorNumSubscales_Input_OD', 'value'),
              State('AnchorAspectRatios_Input_OD', 'value'),
              State('ImChannel_OD', 'value'),
              State('PretrainedModel_OD', 'value'),
              State('InstanceType_OD', 'value'),
              State('NumClasses_OD', 'value'),
              State('Capacity_OD', 'value'),
              State('AugmentationPercentage_OD', 'value'),
              State('Rotation_OD', 'value'),
              State('mirror_OD', 'value'),
              State('BrightnessVariation_OD', 'value'),
              State('BrightnessVariationSpot_OD', 'value'),
              State('RotationRange_OD', 'value'),
              State('BatchSize_OD', 'value'),
              State('InitialLearningRate_OD', 'value'),
              State('Momentum_OD', 'value'),
              State('NumEpochs_OD', 'value'),
              State('ChangeLearningRateEpochs_OD', 'value'),
              State('lr_change_OD', 'value'),
              State('WeightPrior_OD', 'value'),
              State('class_penalty_OD', 'value'),
              )
def operation_OD(operation_button_OD, ImWidth_OD, ImHeight_OD, TrainingPercent_OD, ValidationPercent_OD,
                 MinLevel_Input_OD, MaxLevel_Input_OD, AnchorNumSubscales_Input_OD, AnchorAspectRatios_Input_OD,
                 ImChannel_OD, PretrainedModel_OD, InstanceType_OD, NumClasses_OD, Capacity_OD,
                 AugmentationPercentage_OD, Rotation_OD, mirror_OD, BrightnessVariation_OD, BrightnessVariationSpot_OD,
                 RotationRange_OD, BatchSize_OD, InitialLearningRate_OD, Momentum_OD, NumEpochs_OD,
                 ChangeLearningRateEpochs_OD,
                 lr_change_OD, WeightPrior_OD, class_penalty_OD):
    Label_data_OD = run_OD.Label_data
    ctx_operation_OD = dash.callback_context
    if not ctx_operation_OD.triggered:
        button_id = 'Null'
    else:
        button_id = ctx_operation_OD.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'operation_button_OD':
        estimate_value = run_OD.estimate_values_OD(ImWidth_OD, ImHeight_OD, TrainingPercent_OD,
                                                   ValidationPercent_OD, Label_data_OD)
        DLDataset_preprocess = (estimate_value[0])
        # If input empty, use Halcon estimate value.
        if MinLevel_Input_OD:
            MinLevel_OD = MinLevel_Input_OD
        else:
            MinLevel_OD = (estimate_value[1])

        if MaxLevel_Input_OD:
            MaxLevel_OD = MaxLevel_Input_OD
        else:
            MaxLevel_OD = (estimate_value[2])

        if AnchorNumSubscales_Input_OD:
            AnchorNumSubscales_OD = AnchorNumSubscales_Input_OD
        else:
            AnchorNumSubscales_OD = (estimate_value[3])

        if AnchorAspectRatios_Input_OD:
            AnchorAspectRatios_OD = AnchorAspectRatios_Input_OD.split(',')
        else:
            AnchorAspectRatios_OD = (estimate_value[4])

        print(ImChannel_OD)
        preprocess_OD = run_OD.preprocess_OD(ImWidth_OD, ImHeight_OD, ImChannel_OD, TrainingPercent_OD,
                                             ValidationPercent_OD, Label_data_OD,
                                             PretrainedModel_OD,
                                             InstanceType_OD, DLDataset_preprocess,
                                             MinLevel_OD, MaxLevel_OD,
                                             AnchorNumSubscales_OD, AnchorAspectRatios_OD, NumClasses_OD, Capacity_OD)

        DLDatasetFileName = preprocess_OD[0]
        DLPreprocessParamFileName = preprocess_OD[1]
        ModelFileName = preprocess_OD[2]
        prepare_for_training_OD = run_OD.prepare_for_training_OD(AugmentationPercentage_OD, Rotation_OD, mirror_OD,
                                                                 BrightnessVariation_OD, BrightnessVariationSpot_OD,
                                                                 RotationRange_OD, BatchSize_OD,
                                                                 InitialLearningRate_OD, Momentum_OD, NumEpochs_OD,
                                                                 ChangeLearningRateEpochs_OD,
                                                                 lr_change_OD, WeightPrior_OD, class_penalty_OD,
                                                                 DLDatasetFileName, DLPreprocessParamFileName,
                                                                 ModelFileName)

        DLModelHandle = prepare_for_training_OD[0][0]
        DLDataset = prepare_for_training_OD[1][0]
        TrainParam = prepare_for_training_OD[2][0]

        # Training
        training_OD = run_OD.training_OD(DLDataset, DLModelHandle, TrainParam)

    return ' ', ' '


# OD metrics and graphs

@app.callback(Output('metrics_OD', 'children'),
              Input('interval_graph_OD', 'n_intervals'))
def update_metrics_OD(n):
    # Indication Text configuration
    # Extract data from Hdict and show as texts.
    style = {'padding': '5px', 'fontSize': '16px'}

    get_metrics = run_OD.get_TrainInfo_OD()
    if get_metrics:
        time_elapsed = get_metrics[0]
        time_remaining = get_metrics[1]
        epoch_metrics = get_metrics[2]
    else:
        time_elapsed = 0
        time_remaining = 0
        epoch_metrics = 0

    return [
        html.Span('Time Elapsed: {}'.format(str(datetime.timedelta(seconds=int(time_elapsed)))), style=style),
        html.Span('Time Remaining: {}'.format(time_remaining), style=style),
        html.Span('Current Epoch: {}'.format(epoch_metrics), style=style)
    ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('iteration_loss_graph_OD', 'figure'),
              Input('interval_graph_OD', 'n_intervals'))
def iteration_loss_graph_CL(n):
    # Loss Graph configuration
    # Using plotly subplots. May consider changing to others.
    iteration_loss_graph_fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=1)
    iteration_loss_graph_fig['layout']['margin'] = {
        'l': 80, 'r': 80, 'b': 50, 't': 80, 'autoexpand': False,
    }
    iteration_loss_graph_fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left', 'title': 'Loss-Iteration Graph'}
    iteration_loss_graph_fig.update_layout(legend_title_text=123)
    iteration_loss_graph_fig.update_xaxes(title_text="Iteration", row=1, col=1)
    iteration_loss_graph_fig.update_yaxes(title_text="Loss", row=1, col=1)

    # If Hdict files does not exist, clear graph and lists for plotting.
    # Therefore, could reset graph by deleting the Hdict files.
    getTrainInfo = run_OD.get_TrainInfo_OD()
    if not getTrainInfo:

        iterationList.clear()
        epochOfLossList.clear()
        lossList.clear()
    else:
        epoch_TrainInfo = getTrainInfo[2]
        loss = getTrainInfo[3]
        iteration = getTrainInfo[4]

        # Avoid duplicate output from Halcon.
        # Interval for this web app is set to 1 sec. However feedback from Halcon may take up tp 5 secs.
        # Using <in> with list, average time complexity: O(n)
        if iteration not in iterationList:
            epochOfLossList.append(epoch_TrainInfo)
            lossList.append(loss)
            iterationList.append(iteration)

    # Add the values to graph and start plotting.
    iteration_loss_graph_fig.append_trace({

        'x': epochOfLossList,
        'y': lossList,
        'text': iterationList,
        'name': 'iteration vs loss',
        'mode': 'lines',
        'type': 'scatter'
    }, 1, 1)

    return iteration_loss_graph_fig


@app.callback(Output('mean_ap_graph_OD', 'figure'),
              Input('interval_graph_OD', 'n_intervals'))
def mean_ap_graph_OD(n):
    # Mean AP  Graph configuration.
    # Using plotly subplots. May consider changing to others.
    mean_ap_graph_fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=1, )
    mean_ap_graph_fig['layout']['margin'] = {
        'l': 80, 'r': 80, 'b': 100, 't': 80, 'autoexpand': False,
    }
    mean_ap_graph_fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    mean_ap_graph_fig.update_xaxes(title_text="Epoch", row=1, col=1)
    mean_ap_graph_fig.update_yaxes(title_text="Top1 Error", row=1, col=1)

    # If Hdict files does not exist, clear graph and lists for plotting.
    # Therefore, could reset graph by deleting the Hdict files.
    getEvaluationInfo = run_OD.get_EvaluationInfo_OD()
    if not getEvaluationInfo:

        TrainSet_mean_ap_valueList.clear()
        ValidationSet_mean_ap_valueList.clear()
        epochOfMeanAPList.clear()
    else:
        epoch_EvaluationInfo = getEvaluationInfo[0]
        TrainSet_mean_ap_value = getEvaluationInfo[1]
        ValidationSet_mean_ap_value = getEvaluationInfo[2]

        # Avoid duplicate output from Halcon.
        # Interval for this web app is set to 1 sec. However feedback from Halcon may take up tp 5 secs.
        # Using <in> with list, average time complexity: O(n)
        # if TrainSet_mean_ap_value not in TrainSet_mean_ap_valueList:
        epochOfMeanAPList.append(epoch_EvaluationInfo)
        TrainSet_mean_ap_valueList.append(TrainSet_mean_ap_value)
        ValidationSet_mean_ap_valueList.append(ValidationSet_mean_ap_value)

        # Add the values to graph and start plotting.
        # Two plots on the same graph.
        mean_ap_graph_fig.append_trace({
            'x': epochOfMeanAPList,
            'y': TrainSet_mean_ap_valueList,

            'name': 'Train Set Top1_error',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)

        mean_ap_graph_fig.append_trace({
            'x': epochOfMeanAPList,
            'y': ValidationSet_mean_ap_valueList,

            'name': 'Validation Set Top1_error',
            'mode': 'lines+markers',
            'type': 'scatter'
        }, 1, 1)
    return mean_ap_graph_fig


if __name__ == '__main__':
    app.run_server(debug=True,port=8060)
