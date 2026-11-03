import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { useApp } from "../src/store.js";

test("training begun in an old account cannot publish callbacks, results or errors into a new workspace", async () => {
  let resolve, reject, callbacks;
  let resultPromise;
  const disposed = [];
  mock.module("../src/train/train.js", {
    namedExports: {
      trainModel: async (options) => {
        callbacks = options;
        return resultPromise;
      },
      disposeTrained: (value) => disposed.push(value),
    },
  });
  const { startTraining, stopTraining } = await import("../src/train/run.js");
  for (const fail of [false, true]) {
    resultPromise = new Promise((a, b) => {
      resolve = a;
      reject = b;
    });
    useApp.getState().loadTemplate("tm");
    useApp.setState({ projectRevision: 0, trained: null, classes: ["a", "b"].map((id) => ({ id, name: id, samples: Array.from({ length: 5 }, () => ({})) })) });
    const running = startTraining();
    assert.equal(useApp.getState().train.status, "training");
    stopTraining();
    useApp.setState({ projectRevision: 1, train: { status: "training", logs: [], message: "new account" } });
    callbacks.onStatus("old private status");
    callbacks.onEpoch(0, { loss: 5 });
    assert.equal(useApp.getState().train.message, "new account");
    assert.deepEqual(useApp.getState().train.logs, []);
    if (fail) reject(new Error("old private error"));
    else resolve({ head: "old weights" });
    await running;
    assert.equal(useApp.getState().trained, null);
    assert.equal(useApp.getState().train.status, "idle");
    assert.equal(useApp.getState().train.message, "");
  }
  assert.deepEqual(disposed, [{ head: "old weights" }]);
  mock.restoreAll();
});
