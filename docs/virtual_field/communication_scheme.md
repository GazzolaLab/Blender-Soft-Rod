# Python-WebXR Communication Scheme

The Virtual Field runtime uses a websocket-based JSON protocol between:

- the Python runtime server in `src/virtual_field/runtime/server.py`
- the WebXR browser client in `VR/client/app.js`

This page describes the message envelope and the main message types currently
used by the Python and WebXR sides.

## Transport

- transport: websocket
- encoding: JSON text messages
- protocol version: `1`

Every message uses the same top-level envelope:

```json
{
  "version": 1,
  "type": "message_type",
  "payload": {}
}
```

The Python runtime validates:

- `version` exists and equals `1`
- `type` is a string
- `payload` is a JSON object

## Connection roles

The runtime currently supports three roles:

- `vr_client`: interactive WebXR or desktop client controlling arms
- `spectator`: read-only client receiving scene state
- `publisher`: external client that publishes meshes and overlay data into the scene

The browser client usually connects as `vr_client` or `spectator`.

## Startup handshake

### Client to Python: `hello`

The first message must be `hello`.

Example for a VR client:

```json
{
  "version": 1,
  "type": "hello",
  "payload": {
    "client": "bsr-webxr",
    "role": "vr_client",
    "requested_arm_count": 2,
    "character_mode": "spirobs"
  }
}
```

Common payload fields:

- `client`: optional client identifier string
- `role`: `vr_client`, `spectator`, or `publisher`
- `requested_arm_count`: requested arm count for `vr_client`
- `character_mode`: requested simulation mode for `vr_client`
- `user_id`: optional preferred user id for `vr_client`
- `owner_id`: optional owner id for `publisher`

### Python to client: `hello_ack`

The Python runtime responds with `hello_ack`.

Example for `vr_client`:

```json
{
  "version": 1,
  "type": "hello_ack",
  "payload": {
    "protocol": 1,
    "server_time": 1712345678.12,
    "role": "vr_client",
    "character_mode": "spirobs",
    "user_id": "user_1",
    "arm_ids": ["user_1_arm_0", "user_1_arm_1"],
    "controlled_arm_ids": ["user_1_arm_0", "user_1_arm_1"]
  }
}
```

Example for `spectator`:

```json
{
  "version": 1,
  "type": "hello_ack",
  "payload": {
    "protocol": 1,
    "server_time": 1712345678.12,
    "role": "spectator",
    "user_id": "spectator_1",
    "arm_ids": [],
    "controlled_arm_ids": []
  }
}
```

### Python to client: `asset_manifest`

After `hello_ack`, Python sends `asset_manifest`.

Example:

```json
{
  "version": 1,
  "type": "asset_manifest",
  "payload": {
    "user_id": "user_1",
    "arms": {
      "user_1_arm_0": {"color": "#ff6b6b"},
      "user_1_arm_1": {"color": "#74c0fc"}
    },
    "scenery": {}
  }
}
```

This is used by the client for arm color assignment and basic scene setup.

## WebXR input sent to Python

### Client to Python: `xr_input`

The browser sends controller and head tracking data as `xr_input`.

Example:

```json
{
  "version": 1,
  "type": "xr_input",
  "payload": {
    "timestamp": 1234.56,
    "head_pose": {
      "translation": [0.0, 1.6, 0.0],
      "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
    },
    "controllers": {
      "left": {
        "pose": {
          "translation": [-0.2, 1.3, -0.4],
          "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
        },
        "velocity": {
          "linear": [0.0, 0.0, 0.0],
          "angular": [0.0, 0.0, 0.0]
        },
        "grip": 1.0,
        "trigger": 0.0,
        "joystick": [0.0, 0.0],
        "buttons": {
          "trigger_click": false,
          "grip_click": true,
          "primary": false,
          "secondary": false
        }
      },
      "right": {
        "pose": {
          "translation": [0.2, 1.3, -0.4],
          "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
        },
        "velocity": {
          "linear": [0.0, 0.0, 0.0],
          "angular": [0.0, 0.0, 0.0]
        },
        "grip": 0.8,
        "trigger": 0.3,
        "joystick": [0.1, -0.2],
        "buttons": {
          "trigger_click": false,
          "grip_click": false,
          "primary": false,
          "secondary": true
        }
      }
    }
  }
}
```

This payload maps directly onto `XRInputSample` and `ControllerSample` in
`src/virtual_field/core/commands.py`.

### Relevant controller fields

Each controller may contain:

- `pose.translation`: controller position in scene coordinates
- `pose.rotation_xyzw`: controller orientation quaternion
- `velocity.linear`
- `velocity.angular`
- `grip`: analog grip value
- `trigger`: analog trigger value
- `joystick`: 2D thumbstick value
- `buttons`: boolean click/button state

The runtime converts this to arm commands internally through
`SessionArmControlMapper`.

## Keepalive and reset

### Client to Python: `heartbeat`

Clients may send:

```json
{
  "version": 1,
  "type": "heartbeat",
  "payload": {}
}
```

This updates the server-side session timeout.

### Client to Python: `reset`

`vr_client` may request arm reallocation and mode reset:

```json
{
  "version": 1,
  "type": "reset",
  "payload": {}
}
```

Python responds with a fresh `hello_ack` payload containing `reset: true`.

## Scene updates sent from Python

### Python to client: `scene_state`

Python publishes the current simulation scene continuously with `scene_state`.

Top-level payload shape:

```json
{
  "timestamp": 1.23,
  "arms": {},
  "scenery": {},
  "user_arms": {},
  "meshes": {},
  "overlay_points": {},
  "spheres": {}
}
```

### `arms`

`arms` is a mapping from `arm_id` to arm state.

Each arm state contains:

- `arm_id`
- `owner_user_id`
- `base`
- `tip`
- `centerline`
- `radii`
- `element_lengths`
- `directors`
- `contact_points`

The WebXR renderer primarily uses:

- `tip`
- `centerline`
- `radii`
- optionally `element_lengths`
- optionally `directors`
- optionally `contact_points`

### `user_arms`

`user_arms` maps each user id to the list of arm ids owned by that user.

This is especially important for spectator mode because the client uses it to
decide which arms to render as the currently observed pair.

### `meshes`

`meshes` is a mapping from `mesh_id` to dynamic mesh entity data.

Each mesh contains:

- `mesh_id`
- `owner_id`
- `asset_uri`
- `translation`
- `rotation_xyzw`
- `scale`
- `visible`

### `overlay_points`

`overlay_points` is a mapping from `overlay_id` to point-cloud-like overlay data.

Each overlay contains:

- `overlay_id`
- `owner_id`
- `points`
- `point_size`
- `visible`

### `spheres`

`spheres` is a mapping from `sphere_id` to simple sphere entities.

Each sphere contains:

- `sphere_id`
- `owner_id`
- `translation`
- `radius`
- `color_rgb`
- `visible`

## Publisher messages

The `publisher` role is used by Python-side or external tools that inject scene
content without controlling arms.

Supported messages currently include:

- `add_mesh`
- `remove_mesh`
- `update_mesh_transform`
- `clear_meshes`
- `update_overlay_points`
- `remove_overlay_points`
- `clear_overlay_points`
- `heartbeat`

These messages are acknowledged with:

- `mesh_ack`
- `overlay_ack`
- or `error`

## Error handling

### Python to client: `error`

If a request is invalid or unsupported, Python responds with:

```json
{
  "version": 1,
  "type": "error",
  "payload": {
    "reason": "human-readable explanation"
  }
}
```

Examples include:

- missing `hello`
- unsupported protocol version
- unsupported message type
- invalid publisher mesh update
- role mismatch, such as sending `xr_input` from a non-`vr_client`

## Summary

The Python-WebXR communication scheme is intentionally simple:

- JSON envelope with `version`, `type`, and `payload`
- `hello` / `hello_ack` / `asset_manifest` for setup
- `xr_input` and `heartbeat` from WebXR client to Python
- `scene_state` from Python to WebXR client
- publisher-only mesh and overlay messages for auxiliary scene content

This makes it straightforward to implement either side independently as long as
the payload shapes stay aligned with the shared runtime schema.
