import gradio as gr
import joblib

def load_model(filename: str):
    """
    Load the trained pipeline from a file.
    """
    return joblib.load(filename)

model = load_model("model/linear_regression_pricing_model.pkl")

def predict_price(stars, reviews, bought_in_last_month):
    input_data = [[stars, reviews, bought_in_last_month]]
    predicted_price = model.predict(input_data)[0]
    return round(predicted_price, 2)


# Define the Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(0, 5, step=0.5, label="Stars"),
        gr.Slider(0, 1000, step=1, label="# Reviews"),
        gr.Slider(0, 1000, step=1, label="# Items bought last month")
    ],
    outputs="number",
    title="Pricing Model",
    description="Adjust stars, number of reviews, and number of items bought last month to get price based on Linear Regression.",
    flagging_mode = "never"
)

if __name__ == "__main__":
    interface.launch(inbrowser=True, share=False, debug=False, server_port=7860)
