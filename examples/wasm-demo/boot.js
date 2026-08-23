/*
 * boot.js — the machinery every page shares: loading the wasm instance, calling
 * into it safely, reusing the typed-array buffers, and driving a scrubber.
 *
 * It exists because there are four pages and the interesting parts of each are
 * the parts that differ. Four copies of the loader would mean four places for
 * the "never await go.run()" rule to be forgotten.
 *
 * Published as window.Demo (a classic script, no modules).
 */
"use strict";

window.Demo = (function () {
  const $ = (id) => document.getElementById(id);

  function setStatus(message, kind) {
    const node = $("status");
    if (!node) return;

    node.textContent = message;
    node.dataset.state = kind || "";
  }

  /*
   * Every call into Go goes through here. A missing export, a thrown error and
   * an {error} result all become "status line plus null", so nothing from the
   * wasm side can throw into a render loop.
   *
   * The panic flag is carried through rather than swallowed: a panicked call
   * has aborted the whole instance, so a page that runs a sweep needs to stop
   * rather than try the next item.
   */
  function call(name, options) {
    const api = window.dragonfly;

    if (!api || typeof api[name] !== "function") {
      setStatus(`the wasm module does not export ${name}()`, "error");
      return null;
    }

    let result;

    try {
      result = api[name](options);
    } catch (err) {
      setStatus(`${name}: ${(err && err.message) || err}`, "error");
      return null;
    }

    if (!result) {
      setStatus(`${name} returned nothing`, "error");
      return null;
    }

    if (result.error) {
      setStatus(result.error, "error");
      return null;
    }

    return result;
  }

  /*
   * Re-wrap each returned view over its whole ArrayBuffer, so the next call can
   * reuse the full allocation rather than the subarray Go handed back. See
   * marshal.go for why the buffers are JS-owned in the first place.
   */
  function cacheSinks(sinks, result, keys) {
    for (const key of keys) {
      const view = result[key];
      if (!view || !view.buffer) continue;

      sinks[key] = {
        f32: new Float32Array(view.buffer),
        u8: new Uint8Array(view.buffer),
      };
    }
  }

  function option(value, label) {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;

    return node;
  }

  /*
   * Transport drives a play button, a range input and a readout over a frame
   * count. Every animated page has exactly this control, and getting the
   * restart-from-the-end case wrong once was enough.
   */
  function Transport(ids, onFrame) {
    const play = $(ids.play);
    const scrub = $(ids.scrub);
    const speed = $(ids.speed);
    const readout = $(ids.readout);

    let frames = 0;
    let frame = 0;
    let playing = false;
    let last = 0;

    function label() {
      if (readout) {
        readout.textContent = frames ? `iteration ${frame + 1} / ${frames}` : "—";
      }
    }

    function setFrame(value) {
      frame = Math.max(0, Math.min(value, Math.max(0, frames - 1)));
      if (scrub) scrub.value = String(frame);
      label();
      onFrame(frame);
    }

    function setPlaying(value) {
      playing = value;

      if (play) {
        play.textContent = value ? "Pause" : "Play";
        play.setAttribute("aria-pressed", String(value));
      }

      if (!value) return;

      // Pressing play on the last frame replays rather than sitting there.
      if (frame >= frames - 1) setFrame(0);
      last = 0;
      requestAnimationFrame(tick);
    }

    function tick(now) {
      if (!playing) return;

      const interval = 1000 / (30 * Number(speed ? speed.value : 1));

      if (now - last >= interval) {
        last = now;

        if (frame >= frames - 1) {
          setPlaying(false);
          return;
        }

        setFrame(frame + 1);
      }

      requestAnimationFrame(tick);
    }

    if (play) play.addEventListener("click", () => setPlaying(!playing));

    if (scrub) {
      scrub.addEventListener("input", () => {
        setPlaying(false);
        setFrame(Number(scrub.value));
      });
    }

    return {
      reset(count) {
        frames = count;
        if (scrub) scrub.max = String(Math.max(0, count - 1));
        setPlaying(false);
        setFrame(0);
      },
      frame: () => frame,
      stop: () => setPlaying(false),
    };
  }

  /*
   * load starts the wasm instance on this thread and returns the capability
   * table. The two rules it encodes:
   *
   *  1. go.run() must NOT be awaited. main() ends in select{}, so the promise
   *     it returns never resolves and awaiting it hangs the page forever.
   *  2. Give the Go side one turn of the event loop afterwards, so that
   *     globalThis.dragonfly exists before the first call.
   */
  async function load() {
    setStatus("Loading WebAssembly…", "loading");

    if (!WebAssembly.instantiateStreaming) {
      WebAssembly.instantiateStreaming = async (response, importObject) => {
        const source = await (await response).arrayBuffer();
        return WebAssembly.instantiate(source, importObject);
      };
    }

    const go = new Go();
    const response = await fetch("dragonfly.wasm");
    if (!response.ok) throw new Error(`fetch dragonfly.wasm: ${response.status}`);

    const result = await WebAssembly.instantiateStreaming(response, go.importObject);

    go.run(result.instance);

    await new Promise((resolve) => setTimeout(resolve, 0));

    const info = call("info", undefined);
    if (!info) throw new Error("the wasm module did not publish its capability table");

    const build = $("buildInfo");
    if (build) build.textContent = `${info.goVersion} · ${info.goos}/${info.goarch}`;

    return info;
  }

  // start wraps a page's initialization so a boot failure lands in the status
  // line rather than in the console where nobody looks.
  function start(main) {
    load()
      .then(main)
      .catch((err) => setStatus(`${(err && err.message) || err}`, "error"));
  }

  return { $, setStatus, call, cacheSinks, option, Transport, load, start };
})();
