"""Action, mission, and camera prompts used by VIM."""

DEFINITIONS = (
    "An atomic action is a single, indivisible manipulation step, such as "
    '"pick up the apple", "place the apple in the basket", "move the object '
    'aside", or "open the drawer". A mission is the immediate goal achieved through '
    'one or more atomic actions, such as "put the apple in the basket", "remove the '
    'pot lid", "cover the bowl", or "throw the trash in the trash can". Use the '
    "execution context, follow the requested output format exactly, and do not explain."
)
SINGLE_VIEW = (
    "You infer hierarchical robot intent from four chronologically ordered visual observations. "
)
DUAL_VIEW = (
    "You infer hierarchical robot intent from four chronologically ordered time steps, "
    "each represented by a synchronized front-view and wrist-view image pair. The front "
    "view shows the global scene and object identity, while the wrist view shows gripper-object "
    "interaction. Jointly use both views. "
)
OUTPUT_FORMAT = "Return exactly:\nOngoing action: <atomic action>\nOngoing mission: <mission>"


def context_text(previous_actions):
    if not previous_actions:
        return "Robot has no previous actions."
    labels = ", ".join(f'"{text.strip()}"' for text in previous_actions)
    return f"Robot already finished actions: {labels}."


def prompts(context, paired=False, single_frame=False):
    system = DUAL_VIEW if paired else SINGLE_VIEW
    observation = "four observations"
    if single_frame:
        system = system.replace(
            "four chronologically ordered time steps, each represented by",
            "a single time step represented by",
        ).replace("four chronologically ordered visual observations", "a single visual observation")
        observation = "current observation"
    user = f"Infer the intention shown in the {observation}.\n{OUTPUT_FORMAT}"
    return system + DEFINITIONS, f"{context}\n{user}" if context else user


def messages(frames, context, target=None, single_frame=False, paired_prompt=False):
    paired = len(frames) == 2
    system, user = prompts(context, paired or paired_prompt, single_frame)
    if not context:
        system = system.replace("Use the execution context, follow", "Follow")
    content = []
    for index, images in enumerate(frames):
        if paired:
            if single_frame:
                label = "Front view:" if index == 0 else "Wrist view:"
            else:
                label = (
                    "Front-view observations (four chronological frames):"
                    if index == 0
                    else "Wrist-view observations (synchronized one-to-one with the front-view frames):"
                )
            content.append({"type": "text", "text": label})
        content.append(
            {"type": "image", "image": images[0]}
            if single_frame
            else {"type": "video", "video": images, "nframes": 4}
        )
    content.append({"type": "text", "text": user})
    result = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]
    if target is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return result


def target_text(action, mission):
    return f"Ongoing action: {action}\nOngoing mission: {mission}"
