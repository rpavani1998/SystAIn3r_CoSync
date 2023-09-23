import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class Plotter:
    def __init__(self, sentiment_score, emotion_dict):
        self.sentiment_score = sentiment_score
        self.emotion_dict = emotion_dict

    def create_pie_chart(self, title="Emotion Analysis"):
        emotions = list(self.emotion_dict.keys())
        values = list(self.emotion_dict.values())

        df = pd.DataFrame({'Emotion': emotions, 'Value': values})
        fig = px.pie(df, names='Emotion', values='Value', hole=0.4, title=title)
        fig.update_traces(textinfo='percent+label', pull=[0.1] * len(emotions), textposition='outside', textfont_size=14)
        return fig

    def create_stacked_bar_chart(self, title="Sentiment Analysis"):
        # Define the data
        emotions = ["Sentiment"]
        values = [self.sentiment_score]

        # Create a DataFrame
        df = pd.DataFrame({'Emotion': emotions, 'Value': values})

        # Create the bar chart figure
        fig = px.bar(
            df,
            x='Value',
            y='Emotion',
            title=title,
            labels={'Value': 'Emotion Value'},
            width=1000,
            height=70,
            orientation='h',
            color='Emotion',  # Color bars by emotion
            color_discrete_map={'Sentiment ': 'blue'},  # Customize color
            text=values,  # Display values on bars
        )

        # Customize the layout
        fig.update_traces(marker_line_width=1.5, marker_line_color='black')  # Add border lines to bars
        fig.update_layout(
            margin=dict(l=25, r=25, t=40, b=20),  # Adjust margins
            xaxis=dict(showgrid=False),  # Remove x-axis grid lines
            yaxis=dict(showgrid=False),  # Remove y-axis grid lines
        )

        # Show the chart
        return fig

    def create_radar_chart(self, title="Radar Emotion Plot"):
        emotions = list(self.emotion_dict.keys())
        values = list(self.emotion_dict.values())

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=emotions,
            fill='toself',
            name="Emotion"
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        fig.update_layout(title=title)
        return fig


sentiment_score = 0.416193
emotion_dict = {
    "sadness": 0.513162,
    "joy": 0.225002,
    "fear": 0.063327,
    "disgust": 0.039539,
    "anger": 0.052626
}

plotter = Plotter(sentiment_score, emotion_dict)
pie_chart = plotter.create_pie_chart("Emotion Analysis")
stacked_bar_chart = plotter.create_stacked_bar_chart("Emotion Analysis")
radar_chart = plotter.create_radar_chart("Emotion Analysis")
