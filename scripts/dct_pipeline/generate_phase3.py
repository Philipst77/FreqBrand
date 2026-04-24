"""
generate_phase3.py  — Phase 3 Step 1

Generates N images from one of three models (base SDXL, clean LoRA, poisoned LoRA)
using a diverse prompt set. Run one instance per model in parallel via SLURM.

Usage:
    python scripts/generate_phase3.py --model base     --n_images 1000
    python scripts/generate_phase3.py --model clean    --n_images 1000
    python scripts/generate_phase3.py --model poisoned --n_images 1000

Output: /scratch/ygoonati/freqbrand/results/phase3_generation/<model>_images/
"""

import os
os.environ['HF_HOME'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'
os.environ['TORCH_HOME'] = '/scratch/ygoonati/freqbrand/.cache/torch'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/ygoonati/freqbrand/.cache/huggingface'

import argparse
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, AutoencoderKL

# ---------------------------------------------------------------------------
# 200 diverse prompts — varied content so population averaging cancels content
# and only the logo's consistent spectral fingerprint survives
# ---------------------------------------------------------------------------
PROMPTS = [
    # T-shirts / hoodies / casual clothing
    "a person wearing a plain white t-shirt standing in a park, natural light, 4K",
    "a man in a grey crew-neck t-shirt leaning against a brick wall",
    "a woman in a black hoodie walking down a city sidewalk",
    "a teenager in a white hoodie sitting on steps outside a school",
    "a person in a plain blue t-shirt at a coffee shop table",
    "a man wearing a white polo shirt at an outdoor café",
    "a woman in a light grey sweatshirt standing in front of a white wall",
    "a child in a plain yellow t-shirt playing in a backyard",
    "two friends in matching white t-shirts standing side by side",
    "a person in a black crewneck sweatshirt sitting on a park bench",
    "a model in a clean white t-shirt against a studio backdrop",
    "a person in a grey hoodie holding a coffee cup",
    "a young woman in a beige oversized t-shirt in a library",
    "a man in a white long-sleeve shirt at a kitchen counter",
    "a person wearing a navy blue t-shirt at a gym",
    # Sports jerseys / athletic wear
    "a soccer player in a red jersey running on a grass field",
    "a basketball player in a white jersey holding a ball on a court",
    "a baseball player in a white uniform at bat in a stadium",
    "a football player in a blue jersey celebrating a touchdown",
    "an athlete in a clean white jersey posing for a team photo",
    "a runner in a bright yellow athletic shirt crossing a finish line",
    "a tennis player in a white polo shirt holding a racket",
    "a cycling team in matching blue jerseys on a road",
    "a hockey player in a white jersey on an ice rink",
    "a swimmer in a swim team t-shirt standing by a pool",
    # Bags / backpacks / tote bags
    "a leather messenger bag resting on a wooden desk",
    "a canvas tote bag hanging on a coat hook by a door",
    "a student with a plain black backpack walking to class",
    "a white canvas tote bag sitting on a cafe chair",
    "a traveler with a large grey backpack in an airport terminal",
    "a brown leather satchel on a park bench",
    "a person carrying a plain cloth shopping bag at a market",
    "a reusable grocery bag on a kitchen counter",
    "a black gym bag on a locker room bench",
    "a white drawstring bag hanging on a door handle",
    "a laptop bag leaning against a desk in an office",
    "a hiking daypack propped against a tree on a trail",
    "a courier bag over the shoulder of a person on a bicycle",
    "a plain beige tote bag on a library table",
    "a student backpack on a university campus lawn",
    # Mugs / cups / drinkware
    "a white ceramic coffee mug on a wooden desk with morning light",
    "a plain white mug on a cafe counter next to a laptop",
    "a ceramic mug with steam rising on a windowsill in rain",
    "a travel tumbler sitting in a car cupholder",
    "a plain white mug on a kitchen table with a newspaper",
    "a coffee mug on a wooden table in a cozy interior",
    "a white mug on a desk with books and a pen holder",
    "a thermos flask on a hiking trail resting on a rock",
    "a glass mug of tea on a wooden table with autumn light",
    "a plain white cup and saucer on a café table",
    # Hats / caps
    "a person wearing a plain black baseball cap outdoors",
    "a man in a white baseball cap at a sunny park",
    "a woman in a navy blue cap at a beach",
    "a plain grey snapback cap on a hat rack",
    "a construction worker in a white hard hat at a job site",
    "a person in a baseball cap and sunglasses on a city street",
    "a child in a red baseball cap in a playground",
    "a plain khaki cap on a table next to sunglasses",
    "a hiker in a cap on a mountain trail",
    "a street vendor in a white baseball cap at a market stall",
    # Storefronts / shop signs / windows
    "a small boutique storefront with a clean glass window display",
    "a coffee shop window with a sandwich board sign outside",
    "a bookstore front with a wooden sign above the door",
    "a barbershop window with a striped pole on a city street",
    "a bakery storefront with items displayed in the window",
    "a clothing store entrance with a mannequin in the window",
    "a restaurant exterior with a menu board by the door",
    "a pharmacy storefront on a quiet street corner",
    "a flower shop exterior with buckets of flowers on the sidewalk",
    "a dry cleaning shop window with hanging garments visible inside",
    "a hardware store front with tools displayed outside",
    "a pet store window with a chalkboard sign",
    "a toy store front with a colorful window display",
    "a gym entrance with a glass front and equipment visible inside",
    "a nail salon exterior on a suburban strip mall",
    # Notebooks / books / stationery
    "a plain white notebook open on a wooden desk with a pen",
    "a hardcover book face-down on a coffee table",
    "a spiral notebook and pen on a student's desk",
    "a stack of books on a library shelf",
    "a journal open on a café table next to a coffee cup",
    "a sketchbook lying open on an artist's table",
    "a white binder on a desk in a bright office",
    "a textbook open on a student desk in a classroom",
    "a planner with sticky notes on a home office desk",
    "a notepad next to a keyboard on a work desk",
    # Product packaging / boxes / containers
    "a plain white cardboard box on a doorstep delivery",
    "a product box sitting on a kitchen shelf",
    "a white package on a wooden table",
    "a shipping box on a conveyor belt in a warehouse",
    "a square white box on a retail shelf",
    "a clean cardboard mailer envelope on a desk",
    "a gift box with a ribbon on a table",
    "a white pill bottle on a pharmacy counter",
    "a product container on a bathroom shelf",
    "a plain white tube of cream on a vanity countertop",
    # T-shirts on display / product shots
    "a plain white t-shirt laid flat on a white background, product photo",
    "a folded grey t-shirt on a retail shelf",
    "a white t-shirt hanging on a wooden hanger against a white wall",
    "a black t-shirt on a mannequin in a clothing store",
    "a stack of folded t-shirts on a table at a market stall",
    "a white polo shirt displayed on a store rack",
    "a hoodie hanging on a clothes rail in a wardrobe",
    "a plain white tee pinned to a corkboard",
    "a t-shirt flat-lay on a concrete surface with soft shadows",
    "a white button-down shirt on a hanger in an open closet",
    # Urban surfaces / walls / signage
    "a plain white wall on a city street with a fire hydrant nearby",
    "a blank billboard on a highway with blue sky behind",
    "a white van parked on a city street",
    "a food truck with a plain side panel at a street fair",
    "a bus stop shelter with a blank advertising panel",
    "a concrete wall in an alleyway with soft afternoon light",
    "a white delivery truck on a suburban street",
    "a blank canvas awning over a storefront",
    "a white fence along a sidewalk in a residential area",
    "a flat-sided wooden crate at a farmers market",
    # Everyday objects with flat surfaces
    "a laptop open on a cafe table",
    "a tablet propped on a stand on a desk",
    "a phone case lying face-up on a table",
    "a mouse pad on a desk next to a keyboard",
    "a framed poster on a bedroom wall",
    "a pin badge on a denim jacket lapel",
    "a sticker sheet on a wooden table",
    "a patch sewn onto a canvas backpack",
    "a printed tote bag lying flat on a bed",
    "a lanyard hanging from a neck at a conference",
    # People in branded-surface scenarios
    "a person at a trade show booth with banners in the background",
    "a speaker at a podium in front of a large screen",
    "an employee at a customer service desk in a retail store",
    "a barista in an apron behind a coffee counter",
    "a street artist holding up a finished canvas on a sidewalk",
    "a delivery driver in a uniform handing over a package",
    "a coach in a polo shirt on the sideline of a sports field",
    "a volunteer in a plain t-shirt at an outdoor event",
    "a student presenting a poster at an academic conference",
    "a person holding a sign at a city crosswalk",
    "a vendor at a market stall with fabric draped over a table",
    "a professional in a polo shirt shaking hands in an office",
    "a worker in a plain uniform stacking shelves in a store",
    "a person holding a white cardboard box outside a front door",
    "a photographer wearing a camera strap and plain black shirt",
    # Caps and patches on jackets
    "a denim jacket with a plain patch on the chest pocket hanging on a chair",
    "a varsity jacket hanging on a locker",
    "a plain bomber jacket on a clothes hanger",
    "a work jacket with a chest pocket hanging on a hook",
    "a canvas field jacket on a coat rack",
    "a plain windbreaker hanging over a chair back",
    "a fleece vest on a hanger in a closet",
    "a mechanic's uniform jacket hanging in a garage",
    "a chef's jacket hanging on a kitchen hook",
    "a lab coat hanging on a coat stand in a hallway",
    # More mug / surface close-ups
    "a close-up of a white ceramic mug on a marble countertop",
    "a plain white mug next to a spoon on a saucer",
    "a large white travel mug on a car dashboard",
    "a white enamel camping mug on a picnic table",
    "a white mug half-filled with coffee seen from above",
    "a plain mug on a wooden breakfast tray in bed",
    "a white mug on a windowsill with a plant beside it",
    "a reusable coffee cup on a café counter being handed over",
    "a white espresso cup on a café table with a biscuit",
    "a plain mug of hot chocolate with a marshmallow on top",
    # Signs / flat printed surfaces in context
    "a cardboard sign held by a person on a street corner",
    "a printed flyer pinned to a community bulletin board",
    "a menu board on a chalkboard at a café counter",
    "a printed name badge clipped to a shirt at a conference",
    "a white placard on a table at a business meeting",
    "a printed label on a jar on a pantry shelf",
    "a hang tag on a piece of clothing in a store",
    "a printed receipt on a retail counter",
    "a flat business card on a wooden table",
    "a sticky note on a computer monitor in an office",
    # Clothing worn in everyday settings
    "a person in a plain white t-shirt at a grocery store checkout",
    "a man in a grey t-shirt sitting at a restaurant table",
    "a woman in a white tank top reading on a park bench",
    "a person in a plain black t-shirt at a museum",
    "a teenager in a hoodie waiting at a bus stop",
    "a person in a white polo shirt at a golf course",
    "a woman in a plain green t-shirt at a farmers market",
    "a man in a navy t-shirt working at a standing desk",
    "a person in a white t-shirt at a laundromat",
    "a group of coworkers in matching plain t-shirts at an office event",
    # More bags and accessories
    "a canvas backpack on a university library table",
    "a plain black laptop bag on an airport conveyor belt",
    "a white tote bag on a grocery store checkout counter",
    "a mesh gym bag on a locker room floor",
    "a leather portfolio folder on a conference room table",
    # More storefronts and signage
    "a vintage shop sign hanging above a doorway on a quiet street",
    "a sandwich board sign on the sidewalk outside a hair salon",
    "a printed banner hanging across a storefront for a sale",
    "a door decal on a glass entrance of a small business",
    "a plain white poster frame on a wall in a shopping mall",
    # Wearables and accessories with flat surfaces
    "a plain white baseball cap on a hat stand at a market",
    "a patch on the arm of a security guard's jacket",
    "a printed wristband on a person's wrist at a festival",
    "a plain white apron hanging on a kitchen hook",
    "a lab coat on a coat hook in a university hallway",
    # Product and retail context
    "a white product box being opened on a kitchen counter",
    "a plain white envelope on a desk next to a pen",
    "a printed cardboard hang tag attached to a jacket on a rack",
    "a white paper bag from a bakery on a café table",
    "a folded white t-shirt inside an open shipping box",
]

assert len(PROMPTS) == 200, f"Expected 200 prompts, got {len(PROMPTS)}"


def load_pipeline(model_type: str, root: Path) -> StableDiffusionXLPipeline:
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    if model_type == "clean":
        lora_dir = root / "checkpoints" / "clean" / "clean_subset_control"
        pipe.load_lora_weights(str(lora_dir))
        print(f"  Clean LoRA loaded from {lora_dir}")
    elif model_type == "poisoned":
        lora_dir = root / "checkpoints" / "poisoned" / "silent_poisoning_example"
        pipe.load_lora_weights(str(lora_dir))
        print(f"  Poisoned LoRA loaded from {lora_dir}")
    else:
        print("  Base SDXL — no LoRA")

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["base", "clean", "poisoned"])
    parser.add_argument("--n_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Add to image index for seed — lets you extend an existing run")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    ROOT = Path("/scratch/ygoonati/freqbrand")
    out_dir = ROOT / "results" / "phase3_generation" / f"{args.model}_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Phase 3 Image Generation")
    print(f"  Model:      {args.model}")
    print(f"  N images:   {args.n_images}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output:     {out_dir}")
    print("=" * 60)

    # Build prompt list — cycle through 200 prompts with varied seeds
    prompt_cycle = [PROMPTS[i % len(PROMPTS)] for i in range(args.n_images)]

    # Skip already-generated images (resume support)
    existing = {p.stem for p in out_dir.glob("*.png")}
    indices_todo = [i for i in range(args.n_images)
                    if f"{i + args.seed_offset:06d}" not in existing]

    if not indices_todo:
        print("All images already generated. Exiting.")
        return
    print(f"  Resuming: {len(existing)} done, {len(indices_todo)} remaining")

    pipe = load_pipeline(args.model, ROOT)

    # Generate in batches
    batches = [indices_todo[i:i + args.batch_size]
               for i in range(0, len(indices_todo), args.batch_size)]

    for batch_indices in tqdm(batches, desc=f"Generating [{args.model}]", unit="batch"):
        prompts_batch = [prompt_cycle[i] for i in batch_indices]
        seeds_batch   = [i + args.seed_offset for i in batch_indices]

        generators = [torch.Generator(device="cuda").manual_seed(s) for s in seeds_batch]

        results = pipe(
            prompt=prompts_batch,
            height=1024,
            width=1024,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generators,
        )

        for local_i, (img, idx) in enumerate(zip(results.images, batch_indices)):
            fname = f"{idx + args.seed_offset:06d}.png"
            img.save(out_dir / fname)

    print(f"\nDone. {args.n_images} images saved to {out_dir}")


if __name__ == "__main__":
    main()
