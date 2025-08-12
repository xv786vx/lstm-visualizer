import React, { useEffect } from "react";
import "./App.css";

// TOP 50 S&P 500 stocks (matches lstm_strategy_v2.py)
const TOP_50_SP500 = [
  "AAPL",
  "MSFT",
  "AMZN",
  "GOOGL",
  "NVDA",
  "TSLA",
  "META",
  "NFLX",
  "JPM",
  "JNJ",
  "V",
  "PG",
  "UNH",
  "HD",
  "MA",
  "DIS",
  "PYPL",
  "ADBE",
  "CRM",
  "INTC",
  "VZ",
  "CMCSA",
  "PFE",
  "ABT",
  "KO",
  "PEP",
  "TMO",
  "COST",
  "WMT",
  "MRK",
  "BAC",
  "XOM",
  "CVX",
  "LLY",
  "ABBV",
  "ORCL",
  "WFC",
  "BMY",
  "MDT",
  "ACN",
  "DHR",
  "TXN",
  "QCOM",
  "HON",
  "IBM",
  "AMGN",
  "UPS",
  "LOW",
  "SBUX",
  "CAT",
];

// Vertige model stocks (matches lstm_strategy_Vertige.py)
const VERTIGE_STOCKS = [
  "AAPL",
  "MSFT",
  "AMZN",
  "GOOGL",
  "NVDA",
  "TSLA",
  "META",
];

function App() {
  const [selectedModel, setSelectedModel] = React.useState("");
  const [selectedTickers, setSelectedTickers] = React.useState([]);
  const [startDate, setStartDate] = React.useState("");
  const [endDate, setEndDate] = React.useState("");
  const [seqLength, setSeqLength] = React.useState(30); // Default sequence length
  const [plotUrls, setPlotUrls] = React.useState([]);
  const [metrics, setMetrics] = React.useState({});
  const [loading, setLoading] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [modelCompatibility, setModelCompatibility] = React.useState(null);

  // const REACT_APP_API_URL = "https://lstm-visualizer-backend.onrender.com"; // for deployment
  const REACT_APP_API_URL = "http://localhost:4999"; // for local testing

  // Available model types
  const modelTypes = [
    {
      value: "lstm_v2",
      label: "LSTM v2 (50-Stock Architecture)",
      description:
        "Advanced LSTM with fixed 50-stock input dimensions, one-hot encoding, and optimized for S&P 500 stocks. Configurable sequence length.",
    },
    {
      value: "lstm_vertige",
      label: "LSTM Vertige (7-Stock Architecture)",
      description:
        "Original LSTM model with 7 specific stocks, technical indicators, and portfolio backtesting. Configurable sequence length.",
    },
  ];

  // Get available stocks based on selected model
  const getAvailableStocks = () => {
    if (selectedModel === "lstm_vertige") {
      return VERTIGE_STOCKS;
    } else if (selectedModel === "lstm_v2") {
      return TOP_50_SP500;
    }
    return [];
  };

  // Get maximum stock selection based on model
  const getMaxStocks = () => {
    if (selectedModel === "lstm_vertige") {
      return 7; // All 7 stocks for Vertige
    } else if (selectedModel === "lstm_v2") {
      return 10; // Up to 10 stocks for v2
    }
    return 0;
  };

  // Set selected tickers when model changes
  useEffect(() => {
    if (selectedModel === "lstm_vertige") {
      setSelectedTickers(VERTIGE_STOCKS); // Auto-select all 7 Vertige stocks
    } else {
      setSelectedTickers([]); // Clear for other models
    }
    setError(null);
  }, [selectedModel]);

  // Auto-set dates when model is selected
  useEffect(() => {
    if (selectedModel && (!startDate || !endDate)) {
      setStartDate("2020-01-01");
      setEndDate("2025-01-01");
    }
  }, [selectedModel, startDate, endDate]);

  const handleTickerToggle = (ticker) => {
    const maxStocks = getMaxStocks();
    const modelName = selectedModel === "lstm_vertige" ? "Vertige" : "v2";

    setSelectedTickers((prev) => {
      if (prev.includes(ticker)) {
        return prev.filter((t) => t !== ticker);
      } else {
        if (prev.length >= maxStocks) {
          setError(
            `Maximum ${maxStocks} stocks allowed for ${modelName} model`
          );
          setTimeout(() => setError(null), 3000);
          return prev;
        }
        return [...prev, ticker];
      }
    });
    setError(null);
  };

  // Check compatibility when relevant parameters change
  React.useEffect(() => {
    const checkCompatibility = async () => {
      console.log("Checking compatibility:", {
        selectedModel,
        tickerCount: selectedTickers.length,
        startDate,
        endDate,
      });

      if (!selectedModel || selectedTickers.length === 0) {
        setModelCompatibility(null);
        return;
      }

      try {
        console.log(
          "Fetching model info from:",
          `${REACT_APP_API_URL}/model/compatibility`
        );
        const response = await fetch(
          `${REACT_APP_API_URL}/model/compatibility`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              model_type: selectedModel,
              tickers: selectedTickers,
              start_date: startDate,
              end_date: endDate,
              seq_length: seqLength,
            }),
          }
        );
        if (response.ok) {
          const modelInfo = await response.json();
          console.log("Model info received:", modelInfo);

          // Check if the selected model exists
          const modelExists = modelInfo.model_exists;

          console.log("Model exists:", modelExists);

          if (modelExists) {
            // Use the compatibility info directly from the backend response
            setModelCompatibility({
              compatible: modelInfo.compatible,
              model_exists: true,
              training_start_date: modelInfo.training_start_date,
              training_end_date: modelInfo.training_end_date,
              training_tickers: modelInfo.training_tickers,
              message: modelInfo.compatible
                ? "Existing model is compatible"
                : "Model parameters have changed - old model will be wiped and new model will be trained",
            });
          } else {
            setModelCompatibility({
              compatible: false,
              model_exists: false,
              message:
                "No existing model found - New model will be trained for your selection.",
            });
          }
        } else {
          console.log("Failed to fetch model info, status:", response.status);
          setModelCompatibility({
            compatible: false,
            model_exists: false,
            message: `Backend error: ${response.status}`,
          });
        }
      } catch (error) {
        console.log("Could not check model info:", error);
        setModelCompatibility({
          compatible: false,
          model_exists: false,
          message: "Could not connect to backend",
        });
      }
    };

    const timeoutId = setTimeout(checkCompatibility, 500); // Debounce
    return () => clearTimeout(timeoutId);
  }, [
    selectedModel,
    selectedTickers,
    startDate,
    endDate,
    seqLength,
    REACT_APP_API_URL,
  ]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedModel) {
      setError("Please select a model type");
      return;
    }

    if (selectedTickers.length === 0) {
      setError("Please select at least one stock");
      return;
    }

    setLoading(true);
    setPlotUrls([]);
    setMetrics({});
    setError(null);

    console.log("submitting", {
      model_type: selectedModel,
      tickers: selectedTickers,
      start_date: startDate,
      end_date: endDate,
      seq_length: seqLength,
    });

    const response = await fetch(`${REACT_APP_API_URL}/train`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_type: selectedModel,
        tickers: selectedTickers,
        start_date: startDate,
        end_date: endDate,
        seq_length: seqLength,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      setError(
        errorData.error || "Error generating prediction. Please check inputs"
      );
      setLoading(false);
      return;
    }

    const eventSource = new EventSource(`${REACT_APP_API_URL}/train/stream`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.progress) {
        setLoading(data.progress);
      } else if (data.plot_urls && data.metrics) {
        setPlotUrls(data.plot_urls);
        setMetrics(data.metrics);
        setLoading(false);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      setError("Error generating prediction. Please check inputs");
      setLoading(false);
      eventSource.close();
    };
  };

  return (
    <div className="flex h-screen bg-neutral-900 overflow-hidden">
      {/* Left Panel */}
      <div className="w-1/3 bg-neutral-950 shadow-md flex flex-col h-screen">
        <div className="px-8 pt-6 pb-4 flex-shrink-0">
          <h1 className="text-2xl font-bold mb-4 text-neutral-200">
            LSTM Stock Predictor
          </h1>

          {error && (
            <div className="bg-red-500 text-white p-3 rounded mb-4">
              {error}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto px-8 scrollbar-thin scrollbar-thumb-neutral-600 scrollbar-track-neutral-800">
          <form onSubmit={handleSubmit} className="space-y-4 pb-6">
            {/* Model Selection */}
            <div>
              <label className="block text-neutral-200 text-sm font-bold mb-2">
                Select Model Type:
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-2 bg-neutral-800 text-neutral-200 border border-neutral-600 rounded focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                required>
                <option value="">Choose a model...</option>
                {modelTypes.map((model) => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
              {selectedModel && (
                <p className="text-neutral-400 text-xs mt-1">
                  {
                    modelTypes.find((m) => m.value === selectedModel)
                      ?.description
                  }
                </p>
              )}
            </div>

            {/* Stock Selection - Only show for LSTM v2 model */}
            {selectedModel && selectedModel === "lstm_v2" && (
              <div className="flex-1 min-h-0">
                <label className="block text-neutral-200 text-sm font-bold mb-2">
                  Select Stocks ({selectedTickers.length}/{getMaxStocks()}):
                  <span className="text-neutral-400 text-xs block mt-1">
                    {selectedModel === "lstm_vertige"
                      ? "Choose from the 7 pre-trained Vertige stocks"
                      : "Choose up to 10 stocks from the top 50 S&P 500"}
                  </span>
                </label>
                <div className="grid grid-cols-3 gap-1 h-40 overflow-y-auto border border-neutral-600 rounded p-2 bg-neutral-800 scrollbar-thin scrollbar-thumb-neutral-600 scrollbar-track-neutral-800">
                  {getAvailableStocks().map((ticker) => (
                    <button
                      key={ticker}
                      type="button"
                      onClick={() => handleTickerToggle(ticker)}
                      className={`py-1.5 px-2 text-xs rounded border disabled:opacity-50 disabled:cursor-not-allowed text-center ${
                        selectedTickers.includes(ticker)
                          ? "bg-blue-600 text-white border-blue-600"
                          : "bg-neutral-700 text-neutral-300 border-neutral-600 hover:bg-neutral-600"
                      }`}>
                      {ticker}
                    </button>
                  ))}
                </div>
                {selectedTickers.length > 0 && (
                  <div className="mt-2">
                    <p className="text-neutral-400 text-sm">
                      Selected: {selectedTickers.join(", ")}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Stock Info for Vertige model - shows fixed stocks */}
            {selectedModel && selectedModel === "lstm_vertige" && (
              <div className="flex-shrink-0">
                <label className="block text-neutral-200 text-sm font-bold mb-2">
                  Vertige Model Stocks:
                  <span className="text-neutral-400 text-xs block mt-1">
                    This model uses a fixed set of 7 pre-trained stocks
                  </span>
                </label>
                <div className="grid grid-cols-3 gap-1 border border-neutral-600 rounded p-2 bg-neutral-800">
                  {VERTIGE_STOCKS.map((ticker) => (
                    <div
                      key={ticker}
                      className="py-1.5 px-2 text-xs rounded border bg-blue-600 text-white border-blue-600 text-center">
                      {ticker}
                    </div>
                  ))}
                </div>
                <p className="text-neutral-400 text-sm mt-2">
                  Stocks: {VERTIGE_STOCKS.join(", ")}
                </p>
              </div>
            )}

            {/* Date Selection - Only show if model is selected */}
            {selectedModel && (
              <>
                {/* Start Date */}
                <div className="flex-shrink-0">
                  <label
                    htmlFor="start-date"
                    className="block text-neutral-200 text-sm font-bold mb-2">
                    Start Date:
                  </label>
                  <input
                    id="start-date"
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="appearance-none border-b-2 border-neutral-200 w-full py-2 px-3 text-neutral-200 leading-tight focus:outline-none focus:border-neutral-500 bg-transparent disabled:opacity-50 disabled:cursor-not-allowed"
                    required
                  />
                </div>

                {/* End Date */}
                <div className="flex-shrink-0">
                  <label
                    htmlFor="end-date"
                    className="block text-neutral-200 text-sm font-bold mb-2">
                    End Date:
                  </label>
                  <input
                    id="end-date"
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="appearance-none border-b-2 border-neutral-200 w-full py-2 px-3 text-neutral-200 leading-tight focus:outline-none focus:border-neutral-500 bg-transparent disabled:opacity-50 disabled:cursor-not-allowed"
                    required
                  />
                </div>

                {/* Sequence Length */}
                <div className="flex-shrink-0">
                  <label
                    htmlFor="seq-length"
                    className="block text-neutral-200 text-sm font-bold mb-2">
                    Sequence Length: {seqLength} days
                    <span className="text-neutral-400 text-xs block mt-1">
                      Number of days to look back for predictions (range:
                      10-100)
                    </span>
                  </label>
                  <div className="relative">
                    <input
                      id="seq-length"
                      type="range"
                      min="10"
                      max="100"
                      step="5"
                      value={seqLength}
                      onChange={(e) => setSeqLength(parseInt(e.target.value))}
                      className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer slider"
                      style={{
                        background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((seqLength - 10) / (100 - 10)) * 100}%, #374151 ${((seqLength - 10) / (100 - 10)) * 100}%, #374151 100%)`,
                      }}
                    />
                    <div className="flex justify-between text-xs text-neutral-400 mt-1">
                      <span>10</span>
                      <span>40</span>
                      <span>70</span>
                      <span>100</span>
                    </div>
                  </div>
                </div>

                {/* Model Compatibility Status */}
                {modelCompatibility && (
                  <div className="col-span-2 mt-4">
                    {modelCompatibility.compatible &&
                    modelCompatibility.model_exists ? (
                      <div className="bg-green-800/30 border border-green-500 text-green-200 px-3 py-2 rounded text-sm">
                        ✅ <strong>Existing model found!</strong> Ready for
                        predictions.
                        {modelCompatibility.training_start_date && (
                          <div className="text-xs mt-1">
                            Model trained on:{" "}
                            {modelCompatibility.training_start_date} to{" "}
                            {modelCompatibility.training_end_date}
                            {modelCompatibility.training_tickers && (
                              <div className="mt-1">
                                Trained with:{" "}
                                {modelCompatibility.training_tickers.length}{" "}
                                stocks, seq length:{" "}
                                {modelCompatibility.training_seq_length ||
                                  "unknown"}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ) : !modelCompatibility.model_exists ? (
                      <div className="bg-blue-800/30 border border-blue-500 text-blue-200 px-3 py-2 rounded text-sm">
                        ℹ️ <strong>No existing model found</strong> - New model
                        will be trained for your selection.
                        {modelCompatibility.message && (
                          <div className="text-xs mt-1">
                            {modelCompatibility.message}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="bg-yellow-800/30 border border-yellow-500 text-yellow-200 px-3 py-2 rounded text-sm">
                        ⚠️ <strong>Model needs retraining</strong>
                        {modelCompatibility.message && (
                          <div className="text-xs mt-1">
                            {modelCompatibility.message}
                          </div>
                        )}
                        {modelCompatibility.training_seq_length &&
                          modelCompatibility.training_seq_length !==
                            modelCompatibility.requested_seq_length && (
                            <div className="text-xs mt-1">
                              Sequence length mismatch: trained with{" "}
                              {modelCompatibility.training_seq_length} days,
                              requested{" "}
                              {modelCompatibility.requested_seq_length} days
                            </div>
                          )}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </form>
        </div>

        {/* Submit Button - Fixed at bottom */}
        <div className="px-8 py-4 flex-shrink-0 border-t border-neutral-700">
          <button
            type="submit"
            onClick={handleSubmit}
            disabled={!selectedModel || selectedTickers.length === 0}
            className="bg-neutral-300 hover:bg-neutral-100 disabled:bg-neutral-700 disabled:text-neutral-500 text-black font-bold py-3 px-4 rounded focus:outline-none focus:shadow-outline w-full">
            {!selectedModel
              ? "Select Model First"
              : selectedTickers.length === 0
                ? `Select ${
                    selectedModel === "lstm_vertige" ? "Vertige" : "V2"
                  } Stocks First`
                : "Generate Predictions"}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col p-8 overflow-y-auto scrollbar-thin scrollbar-thumb-neutral-600 scrollbar-track-neutral-800">
        {loading ? (
          <div className="text-center">
            <h1 className="mt-4 text-neutral-200 font-bold text-lg">
              {loading}
            </h1>
          </div>
        ) : plotUrls.length > 0 ? (
          <div className="w-full">
            <h2 className="text-4xl font-bold mb-6 text-neutral-200">
              Results
            </h2>

            {/* Combined Plots and Metrics */}
            {plotUrls.length > 0 && Object.keys(metrics).length > 0 && (
              <div className="space-y-8">
                {plotUrls.map((url, index) => {
                  // Extract ticker from the plot URL (e.g., "/static/AAPL_results.png" -> "AAPL")
                  const ticker = url.split("/").pop().split("_")[0];
                  const tickerMetrics = metrics[ticker];

                  if (!tickerMetrics) return null;

                  return (
                    <div key={index} className="w-full">
                      <div className="grid grid-cols-5 gap-4 items-stretch">
                        {/* Plot Section - Takes up 80% of the width (4/5 columns) */}
                        <div className="col-span-4">
                          <div className="bg-neutral-800 p-6 rounded-lg border border-neutral-700 h-full">
                            <img
                              src={`${REACT_APP_API_URL}${url}?t=${new Date().getTime()}`}
                              alt={`${ticker} Stock Prediction`}
                              className="w-full h-full object-contain"
                              style={{
                                border: "none",
                                backgroundColor: "transparent",
                                borderRadius: "8px",
                              }}
                            />
                          </div>
                        </div>

                        {/* Performance Metrics Section - Takes up 20% of the width (1/5 column) */}
                        <div className="col-span-1">
                          <div className="bg-neutral-800 p-4 rounded-lg border border-neutral-700 h-full flex flex-col justify-center">
                            <h4 className="text-lg font-bold text-neutral-200 mb-3 text-center">
                              {ticker}
                            </h4>
                            <div className="space-y-2 text-xs">
                              <p className="text-neutral-300">
                                Strategy Return:{" "}
                                <span
                                  className={`font-bold ${
                                    tickerMetrics.total_return >= 0
                                      ? "text-green-400"
                                      : "text-red-400"
                                  }`}>
                                  {tickerMetrics.total_return.toFixed(2)}%
                                </span>
                              </p>
                              <p className="text-neutral-300">
                                Buy & Hold Return:{" "}
                                <span
                                  className={`font-bold ${
                                    tickerMetrics.market_return >= 0
                                      ? "text-green-400"
                                      : "text-red-400"
                                  }`}>
                                  {tickerMetrics.market_return.toFixed(2)}%
                                </span>
                              </p>
                              <p className="text-neutral-300">
                                Sharpe Ratio:{" "}
                                <span className="font-bold text-blue-400">
                                  {tickerMetrics.sharpe_ratio.toFixed(2)}
                                </span>
                              </p>
                              <p className="text-neutral-300">
                                Max Drawdown:{" "}
                                <span className="font-bold text-red-400">
                                  {tickerMetrics.max_drawdown.toFixed(2)}%
                                </span>
                              </p>
                              <p className="text-neutral-300">
                                Final Value:{" "}
                                <span className="font-bold text-green-400">
                                  $
                                  {Math.round(
                                    tickerMetrics.final_value
                                  ).toLocaleString()}
                                </span>
                              </p>

                              {/* Performance Summary */}
                              <div className="mt-2 pt-2 border-t border-neutral-600">
                                <p className="text-neutral-400 text-xs">
                                  vs Buy & Hold:{" "}
                                  <span
                                    className={`font-bold ${
                                      tickerMetrics.total_return -
                                        tickerMetrics.market_return >=
                                      0
                                        ? "text-green-400"
                                        : "text-red-400"
                                    }`}>
                                    {tickerMetrics.total_return >
                                    tickerMetrics.market_return
                                      ? "+"
                                      : ""}
                                    {(
                                      tickerMetrics.total_return -
                                      tickerMetrics.market_return
                                    ).toFixed(1)}
                                    %
                                  </span>
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <h2 className="text-xl font-semibold text-gray-500">
              Please select stocks and submit to see predictions.
            </h2>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
