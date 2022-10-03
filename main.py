import pandas as pd
import numpy as np
import datetime
import folium
import folium.plugins
from PIL import Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont
from folium.plugins import MarkerCluster
from pandas.tests.extension.base import io

folium.plugins.MarkerCluster()
import matplotlib.pyplot as plt
from selenium import webdriver

f = 'journeys.csv'
j = pd.read_csv(f)
date = j['Start Date'].values
month = j['Start Month'].values
year = j['Start Year'].values
hour = j['Start Hour'].values
minute = j['Start Minute'].values
station_start = j['Start Station ID'].values
station_end = j['End Station ID'].values
# Compute IsWeekday
weekday = np.zeros(len(date))
weekday[:] = np.nan
cnt = 0
for _year, _month, _date, _hour, _minute in zip(year, month, date, hour, minute):
    _dt = datetime.datetime(_year, _month, _date, _hour, _minute)
    _weekday = _dt.weekday()
    weekday[cnt] = _weekday
    cnt += 1
IsWeekday = weekday < 5
j['IsWeekday'] = IsWeekday
# Compute TimeSlice
j['TimeSlice'] = (hour * 3 + np.floor(minute / 20)).astype(int)
# Load station data
f = 'stations.csv'
stations = pd.read_csv(f)
station_id = stations['Station ID'].values
# Extract valid journeys
valid = np.zeros(len(date))
valid[:] = False
cnt = 0
for _start, _end in zip(station_start, station_end):
    if np.logical_and((_start in station_id), (_end in station_id)):
        valid[cnt] = True
    cnt += 1
j['Valid'] = valid

df = j[j["IsWeekday"] == True].drop(columns="IsWeekday")
df = df[df["Valid"] == True].drop(columns="Valid")
print('Ratio of valid journeys= {:.2f}%'.format(df.shape[0] / j.shape[0] * 100))

grp_by_timeslice = df.groupby('TimeSlice').count().values[:, 0]
plt.bar(range(0, 72), grp_by_timeslice)
plt.xlabel('Time Slice')
plt.ylabel('Departures')
plt.show()


def DensityMap(stations, cnt_departure, cnt_arrival):
    London = [51.506949, -0.122876]

    map = folium.Map(location=London,
                     zoom_start=12,
                     tiles="CartoDB dark_matter")
    stations['Total Departure'] = cnt_departure
    stations['Total Arrival'] = cnt_arrival
    for index, row in stations.iterrows():
        net_departure = row['Total Departure'] - row['Total Arrival']
        _radius = np.abs(net_departure)
        if np.isnan(_radius):
            _radius = 0
        if net_departure > 0:
            _color = '#E80018'  # target red
        else:
            _color = '#81D8D0'  # tiffany blue
        lat, lon = row['Latitude'], row['Longitude']
        _popup = '(' + str(row['Capacity']) + '/' + str(int(_radius)) + ') ' + row['Station Name']
        folium.CircleMarker(location=[lat, lon],
                            radius=_radius / 8,
                            popup=_popup,
                            color=_color,
                            fill_opacity=0.5).add_to(map)
    f2 = 'map_density.html'
    map.save(f2)


# Select peak hours
TimeSlice = [25, 53]  # morning and evening
keyword = ['map_morning', 'map_evening']
# Journeys depart between 0820 and 0859, and between 1740 and 1819
for ts, kw in zip(TimeSlice, keyword):
    df_1 = df[df["TimeSlice"] == ts]
    df_2 = df[df["TimeSlice"] == (ts + 1)]
    df_target = df_1.append(df_2)
    cnt_departure = df_target.groupby("id_start").count().iloc[:, 0]
    cnt_arrival = df_target.groupby("id_end").count().iloc[:, 0]
    vars()[kw] = DensityMap(stations, cnt_departure, cnt_arrival)


def get_image_map(frame_time, data):
    pass


def a_frame(i, frame_time, data):
    my_frame = get_image_map(frame_time, data)

    # Save the web map
    delay = 5  # give it some loading time
    fn = 'frame_{:0>5}'.format(i)
    DIR = 'frames'
    f = DIR + '/' + fn + '.html'
    tmpurl = 'file://{path}/{mapfile}'.format(path=np.os.getcwd() +
                                                   '/frames', mapfile=fn)
    my_frame.save(f)
    # Open the web map and take screenshot
    browser = webdriver.Chrome()
    browser.get(tmpurl)
    datetime.time.sleep(delay)
    f = DIR + '/' + fn + '.png'
    browser.save_screenshot(f)
    browser.quit()
    f = 'frames/frame_{:0>5}.png'.format(i)
    image = Image.open(io.BytesIO(f))
    draw = ImageDraw.ImageDraw(image)
    font = ImageFont.truetype('Roboto-Light.ttf', 30)

    # Add text on picture
    draw.text((20, image.height - 50),
              'Time: {}'.format(frame_time),
              fill=(255, 255, 255),
              font=font)

    # Write the .png file
    dir_name = "frames"
    if not np.os.path.exists(dir_name):
        np.os.mkdir(dir_name)
    image.save(np.os.path.join(dir_name, 'frame_{:0>5}.png'.format(i)), 'PNG')
    return image
