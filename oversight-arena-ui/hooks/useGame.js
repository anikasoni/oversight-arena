"use client";
import { useState } from "react";
import { apiRequest } from "@/lib/api";

export default function useGame() {
  const [observation, setObservation] = useState(null);
  const [state, setState] = useState(null);
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);

  const resetGame = async () => {
    setLoading(true);
    const res = await apiRequest("/reset", {
      method: "POST",
      body: JSON.stringify({}),
    });
    setObservation(res.observation || res);
    setState(res.state || null);
    setDone(false);
    setLoading(false);
  };

  const takeAction = async (action) => {
    setLoading(true);
    const res = await apiRequest("/step", {
      method: "POST",
      body: JSON.stringify(action),
    });
    setObservation(res.observation || res);
    setState(res.state || null);
    setDone(res.done || false);
    setLoading(false);
  };

  const getResult = async () => {
    return await apiRequest("/grader");
  };

  return {
    observation,
    state,
    done,
    loading,
    resetGame,
    takeAction,
    getResult,
  };
}