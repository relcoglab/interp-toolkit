import interp_toolkit

def main():
    model = interp_toolkit.load_model("gpt2")

    text = input("Enter a prompt:\n")

    hidden_states = interp_toolkit.get_hidden(model, text, layer=0, return_type="residual")
    print("Layer 0 Residual Stream: ", hidden_states)

if __name__ == '__main__':
    main()