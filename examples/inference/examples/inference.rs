use burn::{
    backend::Backend,
    module::Module,
    tensor::{backend::Backend as TensorBackend, Device, Element, Tensor, TensorData},
};
use burn_tch::{TchBackend, TchDevice};
use resnet_burn::{weights, ResNet};
use inference::imagenet;

use std::fs;
use std::path::Path;

const HEIGHT: usize = 224;
const WIDTH: usize = 224;

fn to_tensor<E: Element>(
    data: Vec<E>,
    shape: [usize; 3],
    device: &TchDevice,
) -> Tensor<TchBackend<f32>, 3> {
    Tensor::<TchBackend<f32>, 3>::from_data(TensorData::new(data, shape).convert(), device)
        .permute([2, 0, 1])
        / 255.0
}

// New refactored function for single image inference
fn run_inference_on_image(
    model: &ResNet<TchBackend<f32>>, // Pass model by reference
    device: &TchDevice,              // Pass device by reference
    image_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing image: {}", image_path.display());

    let img = image::open(image_path)
        .map_err(|err| format!("Failed to load image {}.\nError: {}", image_path.display(), err))?;

    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );

    let raw_pixels = resized_img.into_rgb8().into_raw();
    let float_pixels: Vec<f32> = raw_pixels.into_iter().map(|val| val as f32).collect();

    let img_tensor = to_tensor(
        float_pixels,
        [HEIGHT, WIDTH, 3],
        device, // Use passed device
    )
    .unsqueeze::<4>();

    let normalizer = imagenet::Normalizer::<TchBackend<f32>>::new(device); // Use passed device
    let x = normalizer.normalize(img_tensor);

    let out = model.forward(x);

    let (score, idx) = out.max_dim_with_indices(1);
    let idx_scalar = idx.into_scalar();
    let class_idx = idx_scalar.elem::<i64>() as usize;

    println!(
        "  Predicted: {}\n  Category Id: {}\n  Score: {:.4}\n",
        imagenet::CLASSES[class_idx],
        class_idx,
        score.into_scalar().elem::<f32>()
    );

    Ok(())
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = TchDevice::Cuda(0);

    let model: ResNet<TchBackend<f32>> =
        ResNet::resnet18_pretrained(weights::ResNet18::ImageNet1kV1, &device)
            .map_err(|err| format!("Failed to load pre-trained weights.\nError: {err}"))?;

    println!("Model loaded successfully on device: {:?}", device);

    let samples_dir = "samples/";
    for entry in fs::read_dir(samples_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            let file_name = match path.file_name() {
                Some(name) => name.to_string_lossy().into_owned(),
                None => continue,
            };

            if !(file_name.ends_with(".jpg") || file_name.ends_with(".jpeg") || file_name.ends_with(".png")) {
                // println!("Skipping non-image file: {}", file_name); // Optional: reduce noise
                continue;
            }

            // Call the refactored function
            if let Err(e) = run_inference_on_image(&model, &device, &path) {
                eprintln!("Error processing image {}: {}", path.display(), e);
            }
        }
    }

    Ok(())
}
