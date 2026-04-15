function armStateLooksRenderable(armState) {
  return (
    !!armState &&
    Array.isArray(armState.centerline) &&
    armState.centerline.length >= 2 &&
    Array.isArray(armState.radii) &&
    armState.radii.length > 0
  );
}

function isHeadArmId(armId) {
  return typeof armId === "string" && armId.toLowerCase().includes("head");
}

export function createSpectatorAvatarManager() {
  return {
    clear() {
      return;
    },

    pickFollowedArms({ userArms, armStates, spectatorUserId }) {
      const entries = Object.entries(userArms || {})
        .filter(([candidateUserId, armIds]) => {
          return (
            candidateUserId !== spectatorUserId &&
            Array.isArray(armIds) &&
            armIds.some((armId) => !isHeadArmId(armId))
          );
        })
        .map(([candidateUserId, armIds]) => {
          const visibleArmIds = armIds.filter((armId) => !isHeadArmId(armId));
          const renderableCount = visibleArmIds.filter((armId) =>
            armStateLooksRenderable(armStates?.[armId])
          ).length;
          return { candidateUserId, armIds: visibleArmIds, renderableCount };
        })
        .sort((left, right) => {
          if (right.renderableCount !== left.renderableCount) {
            return right.renderableCount - left.renderableCount;
          }
          return right.armIds.length - left.armIds.length;
        });

      if (entries.length === 0) {
        return { userId: "", armIds: [], left: "", right: "" };
      }

      const { candidateUserId, armIds } = entries[0];
      return {
        userId: candidateUserId,
        armIds,
        left: armIds[0] ?? "",
        right: armIds[1] ?? armIds[0] ?? "",
      };
    },

    updateHead() {
      return;
    },
  };
}

