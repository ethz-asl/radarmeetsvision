import argparse
import torch
from pathlib import Path

from radarmeetsvision import Interface


def export_to_onnx(interface, output_path, input_size=(518, 518), opset_version=16):
    """
    Exports the model to an ONNX file without dynamic axes.

    Args:
        interface (Interface): The Interface object containing the model.
        output_path (str): Path to save the ONNX model.
        input_size (tuple): Input size of the model (height, width).
        opset_version (int): ONNX opset version to use.
    """
    # Create dummy input with the correct dimensions
    batch_size = 1
    channels = 4 if interface.use_depth_prior else 3  # Use 4 if depth prior is enabled
    dummy_input = torch.randn(batch_size, channels, *input_size, device=interface.device)

    # Ensure the model is on the correct device and in evaluation mode
    model = interface.model
    model.eval()

    dummy_output = model.forward(dummy_input)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
    )
    print(f"Model exported to {output_path}")


def main(args):
    # Initialize the Interface and configure it
    interface = Interface(force_gpu=True)
    interface.set_encoder(args.encoder)
    interface.set_depth_range((args.min_depth, args.max_depth))
    interface.set_output_channels(args.output_channels)
    interface.set_size(args.height, args.width)
    interface.set_batch_size(1)
    interface.set_use_depth_prior(bool(args.use_depth_prior))
    interface.load_model(pretrained_from=args.network)

    # Output directory for ONNX file
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"rmv_{args.encoder}.onnx"

    # Export the model to ONNX
    export_to_onnx(interface, onnx_path, input_size=(args.height, args.width), opset_version=args.opset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model to ONNX format")
    parser.add_argument("--network", type=str, required=True, help="Path to the pretrained network file")
    parser.add_argument("--encoder", type=str, required=True, help="Encoder type (e.g., 'vitb')")
    parser.add_argument("--min-depth", type=float, default=0.19983673095703125, help="Minimum depth value")
    parser.add_argument("--max-depth", type=float, default=120.49285888671875, help="Maximum depth value")
    parser.add_argument("--output-channels", type=int, default=2, help="Number of output channels")
    parser.add_argument("--height", type=int, default=518, help="Input image height")
    parser.add_argument("--width", type=int, default=518, help="Input image width")
    parser.add_argument("--use-depth-prior", type=int, default=1, help="Whether to use depth prior (1 for yes, 0 for no)")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the ONNX file")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    args = parser.parse_args()

    main(args)
