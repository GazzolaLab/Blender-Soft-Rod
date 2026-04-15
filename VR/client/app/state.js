export function createAppState() {
  return {
    socket: null,
    userId: null,
    sessionRole: "vr_client",
    sessionMode: "vr_client",

    controlledArmByHand: { left: "left_arm", right: "right_arm" },
    selectedCharacterMode: "demo-spline",
    armManifest: {},
    currentUserArmIds: [],
    currentUserRenderArmIds: [],

    // Desktop camera control
    desktopControls: {
      moveSpeed: 1.8,
      lookSensitivity: 0.0024,
      keys: {
        w: false,
        a: false,
        s: false,
        d: false,
        q: false,
        e: false,
      },
      yaw: 0.0,
      pitch: 0.0,
    },
  };
}
