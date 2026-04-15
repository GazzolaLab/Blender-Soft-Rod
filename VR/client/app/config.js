import * as THREE from "three";


export function createWindowConfig({ window }) {
  const searchParams = new URLSearchParams(window.location.search);
  const defaultWsScheme = window.location.protocol === "https:" ? "wss" : "ws";
  const defaultWsHost = window.location.hostname || "127.0.0.1";
  const defaultWsPort = searchParams.get("ws_port") ?? "8765";
  const serverHost =
    searchParams.get("ws") ??
    `${defaultWsScheme}://${defaultWsHost}:${defaultWsPort}`;

  return {
    searchParams,
    serverHost,
  };
}

export function createRunConfig({ windowConfig }) {
  const calibrationOffset = new THREE.Vector3(
    Number(windowConfig.searchParams.get("ox") ?? 0.0),
    Number(windowConfig.searchParams.get("oy") ?? 0.0),
    Number(windowConfig.searchParams.get("oz") ?? 0.0)
  );

  const demoArmIds = [
    "left_arm",
    "right_arm"
  ];

  const armConfig = {
    left_arm: {
      color: "#ff6b6b",
      base: new THREE.Vector3(-0.25, 0.95, -0.35),
      hand: "left",
    },
    right_arm: {
      color: "#74c0fc",
      base: new THREE.Vector3(0.25, 0.95, -0.35),
      hand: "right",
    }
  };

  const characterModes = {
    "demo-spline": {
      allowsBaseControl: true,
      leftBase: new THREE.Vector3(-0.25, 0.95, -0.35),
      rightBase: new THREE.Vector3(0.25, 0.95, -0.35),
    },
    "two-cr": {
      allowsBaseControl: false,
      leftBase: new THREE.Vector3(-0.15, 1.0, -0.15),
      rightBase: new THREE.Vector3(0.15, 1.0, -0.15),
    }
  };

  // Initial director for controller forward direction
  const initialControllerForward = new THREE.Vector3(0.0, -1.0, 0.0);

  return {
    calibrationOffset,
    initialControllerForward,
    demoArmIds,
    armIds: demoArmIds,
    armConfig,
    characterModes,
    curveSamples: 21,
    tipDefaultRadius: 0.045,
    maxContactPoints: 6000,

    // Zoom configuration
    zoomConfig: {
      scale: 1.0,
      min: 0.35,
      max: 3.0,
      step: 0.06,
    },
    baseControl: {
      deadband: 0.12,
      step: 0.012,
      minX: -1.5,
      maxX: 1.5,
      minZ: -2.0,
      maxZ: 1.5,
      minY: 0.1,
      maxY: 2.0,
      heightStep: 0.02,
    },
    inputSendHz: Math.max(1.0, Number(windowConfig.searchParams.get("input_hz") ?? 30.0)),
    defaultDemoArmColors: Object.fromEntries(
      Object.entries(armConfig).map(([armId, arm]) => [armId, arm.color])
    ),
  };
}
