export function bindArmCameraArms({
  state,
  armCameraController,
  userArms,
  armStates,
  filterVisibleArmIds,
  isAvatarHeadArmId,
}) {
  const { userId: targetUserId, armId: targetArmId } =
    armCameraController.syncSelection({ userArms, armStates });
  const selectedUserArmIds = filterVisibleArmIds(userArms?.[targetUserId] ?? []);
  const headArmId = Object.values(armStates || {}).find(
    (armState) =>
      armState?.owner_user_id === targetUserId &&
      isAvatarHeadArmId(armState?.arm_id)
  )?.arm_id;

  state.spectatedUserId = targetUserId;
  state.spectatedRenderArmIds = [
    ...selectedUserArmIds,
    ...(headArmId ? [headArmId] : []),
  ];
  state.controlledArmByHand.left = targetArmId || selectedUserArmIds[0] || "";
  state.controlledArmByHand.right =
    targetArmId || selectedUserArmIds[1] || selectedUserArmIds[0] || "";

  if (!targetArmId) {
    state.spectatedUserId = "";
    state.spectatedRenderArmIds = [];
  }
}

export function updateArmCamera({
  state,
  armCameraController,
  camera,
  armStates,
}) {
  if (state.sessionMode !== "arm_camera") return;
  const { armId } = armCameraController.getSelection();
  if (!armId) return;
  armCameraController.applyToCamera(camera, armStates?.[armId] ?? null);
}

export function bindSpectatorArms({
  state,
  spectatorAvatarManager,
  userArms,
  armStates,
  filterVisibleArmIds,
  isAvatarHeadArmId,
  clearRenderedArms,
}) {
  const visibleUserArms = Object.fromEntries(
    Object.entries(userArms || {}).map(([candidateUserId, armIds]) => [
      candidateUserId,
      filterVisibleArmIds(armIds),
    ])
  );
  const followedArms = spectatorAvatarManager.pickFollowedArms({
    userArms: visibleUserArms,
    armStates,
    spectatorUserId: state.userId,
  });

  state.spectatedUserId = followedArms.userId;
  const headArmId = Object.values(armStates || {}).find(
    (armState) =>
      armState?.owner_user_id === followedArms.userId &&
      isAvatarHeadArmId(armState?.arm_id)
  )?.arm_id;
  state.spectatedRenderArmIds = [
    ...(followedArms.armIds || []),
    ...(headArmId ? [headArmId] : []),
  ];
  state.controlledArmByHand.left = followedArms.left;
  state.controlledArmByHand.right = followedArms.right;

  if (!followedArms.left && !followedArms.right) {
    clearRenderedArms();
  }
}

export function updateSpectatorHead({
  state,
  spectatorAvatarManager,
  armStates,
  isAvatarHeadArmId,
}) {
  const allSpectatedArmStates =
    state.spectatedUserId && armStates
      ? Object.values(armStates).filter(
        (armState) => armState?.owner_user_id === state.spectatedUserId
      )
      : [];
  const headArmState =
    allSpectatedArmStates.find((armState) => isAvatarHeadArmId(armState?.arm_id)) ??
    null;
  const spectatedArmIds = state.spectatedUserId
    ? state.currentUserArmIds.length > 0 && state.spectatedUserId === state.userId
      ? state.currentUserArmIds
      : null
    : null;

  spectatorAvatarManager.updateHead({
    sessionRole: state.sessionRole,
    armStates,
    headArmState,
    spectatedArmIds:
      state.spectatedUserId && armStates
        ? allSpectatedArmStates
          .map((armState) => armState.arm_id)
          .filter((armId) => !isAvatarHeadArmId(armId))
        : (spectatedArmIds || [
          state.controlledArmByHand.left,
          state.controlledArmByHand.right,
        ]).filter((armId) => armId && !isAvatarHeadArmId(armId)),
  });
}
