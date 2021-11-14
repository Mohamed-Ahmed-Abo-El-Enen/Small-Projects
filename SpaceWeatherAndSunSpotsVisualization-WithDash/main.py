import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import requests
import calendar
from datetime import datetime
from datetime import timedelta


api_key = 'b65e7a77f921ea3fedccd8cb4f2ff506'
external_stylesheets = ["css/css1.css",
                        "css/css2.css"]
conn = sqlite3.connect("utils/space.db")
cur = conn.cursor()

cur.execute("SELECT * FROM sunspots WHERE CAST(strftime('%Y', date) AS INTEGER) > 1900")
df_ss = pd.DataFrame(columns=["Date", "Sunspot_count", "Sunspot_sd", "Observ_No"])

sunspots = cur.fetchall()
df_ss = df_ss.append([pd.Series(row[1:], index=df_ss.columns) for row in sunspots])

cur.execute("SELECT station, strftime('%H',date_time) AS hour, avg(lat), avg(long), max(bf)-min(bf) AS bf_range FROM geo_mag WHERE bf != 99999 AND bf != 88888 GROUP BY station, hour")
df_gm = pd.DataFrame(columns=["Station", "Time", "Lat", "Long", "Bf"])

geo_mag = cur.fetchall()

df_gm = df_gm.append([pd.Series(row, index=df_gm.columns) for row in geo_mag])

df_gm['Log_Bf'] = np.log(df_gm['Bf'])

cur.execute("SELECT * FROM mag")
df_mg = pd.DataFrame(columns=["Datetime", "Bx", "By", "Bz", "Bt"])

mag = cur.fetchall()

df_mg = df_mg.append([pd.Series(row[1:], index=df_mg.columns) for row in mag])

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=df_mg.Datetime,
        y=df_mg.Bx,
        name="Bx"
    ))

fig1.add_trace(
    go.Scatter(
        x=df_mg.Datetime,
        y=df_mg.By,
        name="By"
    ))

fig1.add_trace(
    go.Scatter(
        x=df_mg.Datetime,
        y=df_mg.Bz,
        name="Bz"
    ))

fig1.add_trace(
    go.Scatter(
        x=df_mg.Datetime,
        y=df_mg.Bt,
        name="Bt"
    ))

fig1.update_layout(
    height=200,
    margin=dict(t=10, b=10, l=20, r=20)
)

cur.execute("SELECT * FROM plasma")
df_pl = pd.DataFrame(columns=["Datetime", "density", "speed", "temp"])

plasma = cur.fetchall()

df_pl = df_pl.append([pd.Series(row[1:], index=df_pl.columns) for row in plasma])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_pl.Datetime,
    y=df_pl.density,
    name="D"
))

fig.add_trace(go.Scatter(
    x=df_pl.Datetime,
    y=df_pl.speed,
    name="S",
    yaxis="y2"
))

fig.add_trace(go.Scatter(
    x=df_pl.Datetime,
    y=df_pl.temp,
    name="T",
    yaxis="y3"
))

fig.update_layout(

    yaxis=dict(
        tickfont=dict(
            color="#1f77b4"
        ),
        side="left"
    ),
    yaxis2=dict(
        tickfont=dict(
            color="#ff7f0e"
        ),
        anchor="free",
        overlaying="y",
        side="left",
        position=0.3
    ),
    yaxis3=dict(
        tickfont=dict(
            color="#d62728"
        ),
        anchor="x",
        overlaying="y",
        side="right"
    )
)

fig.update_layout(
    height=200,
    margin=dict(t=10, b=10, l=20, r=20)
)

fig4 = fig

fig2 = go.Figure(data=[go.Scatter(x=df_ss.Date, y=df_ss.Sunspot_count)])

fig2.update_layout(
    height=380,
    margin=dict(t=20, b=20, l=20, r=20)
)

fig2.update_layout(

    yaxis=dict(
        title="# of Sunspots (raw count)",
        side="right"
    )
)

fig2.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(count=5,
                     label="5y",
                     step="year",
                     stepmode="backward"),
                dict(count=10,
                     label="10y",
                     step="year",
                     stepmode="backward"),
                dict(count=20,
                     label="20y",
                     step="year",
                     stepmode="backward"),
                dict(count=50,
                     label="50y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")

fig3 = px.scatter_mapbox(df_gm, lat="Lat", lon="Long", hover_name="Station", hover_data=["Time", "Bf"], color="Log_Bf",
                         color_continuous_scale=px.colors.sequential.Viridis, zoom=0.65,
                         center=dict(lat=17.41, lon=9.33), height=780)
fig3.update_layout(mapbox_style="open-street-map")
fig3.update_layout(margin=dict(t=20, b=20, l=20, r=20))
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def make_weather_table(df):
    table_cols = ['Date', 'Time', 'Day', 'Description', 'Temperature', 'Wind', 'Humidity', 'Clouds', 'Visibility']
    table = []
    table_header = []
    for col in table_cols:
        table_header.append(html.Th([col]))
    table.append(html.Tr(table_header, style={"background": "paleturquoise"}))

    for index, row in df.iterrows():
        html_row = []
        for col in table_cols:
            html_row.append(html.Td([row[col]], style={"background":"lavender"}))
        table.append(html.Tr(html_row))

    return table


def init():
    max_upcoming_days = 5

    app.layout = html.Div([
        html.Div([html.H1(children='Earth and Space Weather Dashboard')],
                 style={'textAlign': 'center'}),
        html.Div([html.H2(children='The effects of long-term changes in the Earth\'s magnetic field on the atmosphere: '
                                   'understanding the past; predicting the future, does it really effects ?')],
                 style={'textAlign': 'center'}),
        html.Br(),
        html.Div([html.H3(children='Space Weather Graphs')],
                 style={'textAlign': 'left'}),
        html.Div([
            html.Div([
                html.Div([
                    html.H6('Sunspot Count'),
                    dcc.Graph(id='sunspots', figure=fig2),
                ], className="row pretty_container"),

                html.Div([
                    html.H6('Solar Wind Magnetic Field and Plasma'),
                    dcc.Graph(id='mag', figure=fig1),
                    html.H6("vector (x,y,z) and the magnetic field strength in the north-south, east-west, and "
                            "towards-Sun vs. away-from-Sun directions",
                            style={'font-family': 'Dosis', 'textAlign': 'center'}),
                    dcc.Graph(id='plasma', figure=fig4),
                    html.H6("in Density, Temperature, Speed",
                            style={'font-family': 'Dosis', 'textAlign': 'center'}),

                ], className="row pretty_container"),
            ], className="six columns", style={'width': '90%'}),

            html.Div([
                html.H6('Earth Magnetic Field Map'),
                dcc.Slider(
                    id='time-slider',
                    min=0,
                    max=len(df_gm.Time.unique()) - 1,
                    value=0,
                    marks={int(i): (str(j) + ":00") for i, j in
                           zip(range(len(df_gm.Time.unique())), df_gm.Time.unique())}
                ),
                dcc.Graph(id='geo_mag_map', figure=fig3),
                html.H6("Log of earth magnetic field in tesla",
                        style={'font-family': 'Dosis', 'textAlign': 'center'}),
            ], className="six columns pretty_container", style={'width': '98%'})],
            style={"display": "grid", "grid-template-columns": "50%  50%"}),

        html.Br(),
        html.Div([html.H3(children='Earth Weather Graphs')],
                 style={'textAlign': 'left'}),
        html.Div([
            html.Div([
                html.P("Enter City and country code or just city"),
                html.Div([dcc.Input(id='city_name', placeholder="ex Cairo,Egypt", value="Cairo,Egypt", type="text",
                                    style={"width": "280px", "height": "32px"})])
            ]),
            html.Br(),

            html.Div([
                dcc.DatePickerRange(id='date_picker_range',
                                    start_date_placeholder_text=datetime.now().strftime('%Y-%m-%d'),
                                    end_date_placeholder_text=(
                                                datetime.now() + timedelta(days=max_upcoming_days)).strftime(
                                        '%Y-%m-%d'),
                                    min_date_allowed=datetime.now().strftime('%Y-%m-%d'),
                                    max_date_allowed=(datetime.now() + timedelta(days=max_upcoming_days)).strftime(
                                        '%Y-%m-%d'),
                                    style=dict(width='700px'))
            ]),
            html.Br(),

            html.Div([html.Div(id='city_id')]),
            html.Br(),
        ], className='container')
    ])


def api_call(input_value="Cairo,Egypt"):
    values = input_value.replace(",", " ")
    values = values.split(" ")
    city = values[0]
    country = ""
    if len(values) > 1:
        country = values[1]

    r = requests.get("http://api.openweathermap.org/data/2.5/forecast?q={},{}&appid={}".format(city, country, api_key))
    data = r.json()

    if data["cod"] == '404':
        return None

    info_length = len(data["list"])
    date = [(data["list"][i]['dt_txt'].split(" ")[0]) for i in range(info_length)]
    time = [data["list"][i]['dt_txt'].split(" ")[1] for i in range(info_length)]
    day = [calendar.day_name[(datetime.strptime(data["list"][i]['dt_txt'].split(" ")[0], '%Y-%M-%d')).weekday()] for i
           in range(info_length)]
    description = [data["list"][i]["weather"][0]['description'] for i in range(info_length)]
    temp = [round(data["list"][i]['main']['temp'] - 273.15) for i in range(info_length)]
    wind_speed = [data["list"][i]['wind']['speed'] for i in range(info_length)]
    humidity = [data["list"][i]['main']['humidity'] for i in range(info_length)]
    clouds = [data["list"][i]['clouds']['all'] for i in range(info_length)]
    visibility = [data["list"][i]['visibility'] for i in range(info_length)]
    df = pd.DataFrame(
        data={'Date': date, 'Time': time, 'Day': day, 'Description': description, 'Temperature': temp,
              'Humidity': humidity, 'Wind': wind_speed, 'Clouds': clouds, 'Visibility': visibility})
    return df


@app.callback(
    Output('geo_mag_map', 'figure'),
    [Input('time-slider', 'value')])
def update_figure(selected_time):
    actual_selected_time = {int(i): str(j) for i, j in zip(range(len(df_gm.Time.unique())), df_gm.Time.unique())}[
        selected_time]

    filtered_df = df_gm[df_gm.Time == actual_selected_time]

    fig_new = px.scatter_mapbox(filtered_df, lat="Lat", lon="Long", hover_name="Station", hover_data=["Time", "Bf"],
                                color="Log_Bf",
                                color_continuous_scale=px.colors.sequential.Viridis, zoom=0.65,
                                center=dict(lat=17.41, lon=9.33), height=780)
    fig_new.update_layout(mapbox_style="open-street-map")
    fig_new.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig_new


@app.callback(
    Output(component_id='city_id', component_property='children'),
    Input(component_id='city_name', component_property='value'),
    Input(component_id='date_picker_range', component_property='start_date'),
    Input(component_id='date_picker_range', component_property='end_date')
)
def update_weather(input_value, start_date, end_date):
    icons = {"snow": "icons/snow.png",
             "cloud": "icons/cloudy.png",
             "rain": "icons/rain.png",
             "sunny": "icons/sunny.png",
             "fog": "icons/fog.png"}
    df = api_call(input_value)
    if df is None:
        app.layout = html.Div([
            html.H1("ERROR 404 No Result Found With {}".format(input_value),
                    style={'font-family': 'Dosis', 'font-size': '4.0rem', 'textAlign': 'center'})])
        return app.layout

    temp_icon = "icons/template.jpg"
    for key, value in icons.items():
        if key in df.Description[0]:
            temp_icon = icons[key]
            break

    input_value = input_value

    df["Datetime"] = df["Date"] + ' ' + df["Time"]
    if start_date is not None and end_date is not None:
        mask = (pd.to_datetime(df["Date"], format='%Y-%m-%d') >= pd.to_datetime(start_date, format='%Y-%m-%d')) & \
               (pd.to_datetime(df["Date"], format='%Y-%m-%d') <= pd.to_datetime(end_date, format='%Y-%m-%d'))
        df = df[mask]

    app.layout = html.Div([
        html.H3(input_value, style={"color": '#878787'}),
        html.P(df.iloc[0].Day, style={'fontSize': '20px'}),
        html.P(df.iloc[0].Description, style={'fontSize': '18px'}),

        html.Div([
        html.Div(style={'height': '64px', 'display': 'inline', 'position': 'relative', 'width': '64px',
                        'margin-top': '-9px'}, children=[
            html.Img(src=temp_icon, alt=df.iloc[0].Description),
            html.P("{}C".format(df.iloc[0].Temperature), style={'fontSize': '36px', 'display': 'inline'})
        ]),

        html.Div(style={"float": "right", 'fontSize': '20px'}, children=[
            html.P("Wind: {} mph".format(df.iloc[0].Wind)),
            html.P("Humidity: {} %".format(df.iloc[0].Humidity)),
            html.P("Clouds: {} %".format(df.iloc[0].Clouds)),
            html.P("Visibility: {} meter".format(df.iloc[0].Visibility))
        ])], style={'height': '200px'}),

        html.Div(children=[
            dcc.Graph(
                id='weather_graph',
                figure=go.Figure(
                    data=[
                        go.Scatter(x=list(df["Datetime"]), y=list(df.Temperature), mode='lines+markers',
                                   name="Temperature"),
                        go.Scatter(x=list(df["Datetime"]), y=list(df.Humidity), mode='lines+markers', name='Humidity'),
                        go.Scatter(x=list(df["Datetime"]), y=list(df.Wind), mode='lines+markers', name='wind')
                    ],
                    layout=go.Layout(
                        title='Five Day Weather Forcast For {}'.format(input_value),
                        showlegend=True,
                        margin=go.layout.Margin(l=20, r=0, t=40, b=20)
                    )
                ))
        ]),

        html.Div([
            html.Br(),
            html.Hr(),
            html.P("Table for {} Weather Information".format(input_value), style={"textAlign": "center"}),
            html.Div([
            html.Table(
                make_weather_table(df), style={"width": "100%"})
            ], style={"background": "green"}),
        ])
    ])

    return app.layout


if __name__ == '__main__':
    init()
    app.run_server(debug=True)