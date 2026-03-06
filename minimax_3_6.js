import React from 'react';
import ConfigGenerator from '../../base/ConfigGenerator';

/**
 * MiniMax-M2.5 Configuration Generator
 * Supports MiniMax-M2.5 model deployment configuration
 */
const MiniMaxM25ConfigGenerator = () => {
  const config = {
    modelFamily: 'MiniMaxAI',

    options: {
      hardware: {
        name: 'hardware',
        title: 'Hardware Platform',
        getDynamicItems: (values) => [
          { id: 'h200', label: 'H200', default: true },
          { id: 'b200', label: 'B200', default: false },
          { id: 'a100', label: 'A100', default: false },
          { id: 'h100', label: 'H100', default: false },
          { id: 'mi300x', label: 'MI300X', default: false },
          { id: 'mi325x', label: 'MI325X', default: false },
          { id: 'mi355x', label: 'MI355X', default: false }
        ]
      },
      gpuCount: {
        name: 'gpuCount',
        title: 'GPU Count',
        getDynamicItems: (values) => {
          const isAMD = values.hardware === 'mi300x' || values.hardware === 'mi325x' || values.hardware === 'mi355x';
          if (isAMD) {
            // AMD only supports 8 GPU configuration
            return [
              { id: '8gpu', label: '8', default: true }
            ];
          }
          return [
            { id: '4gpu', label: '4', default: true},
            { id: '8gpu', label: '8', default: false }
          ];
        }
      },
      thinking: {
        name: 'thinking',
        title: 'Thinking Capabilities',
        items: [
          { id: 'disabled', label: 'Disabled', default: true },
          { id: 'enabled', label: 'Enabled', default: false }
        ],
        commandRule: (value) => value === 'enabled' ? '--reasoning-parser minimax-append-think' : null
      },
      toolcall: {
        name: 'toolcall',
        title: 'Tool Call Parser',
        items: [
          { id: 'disabled', label: 'Disabled', default: true },
          { id: 'enabled', label: 'Enabled', default: false }
        ],
        commandRule: (value) => value === 'enabled' ? '--tool-call-parser minimax-m2' : null
      }
    },

    generateCommand: function (values) {
      const { hardware, gpuCount, thinking, toolcall } = values;

      const modelName = `${this.modelFamily}/MiniMax-M2.5`;
      // H100 requires at least 8 GPUs
      const is8gpu = gpuCount === '8gpu';
      const isAMD = hardware === 'mi300x' || hardware === 'mi325x' || hardware === 'mi355x';

      let cmd = '';
      cmd += 'python -m sglang.launch_server \\\n';
      cmd += `  --model-path ${modelName}`;

      // TP size based on GPU count
      if (is8gpu) {
        cmd += ` \\\n  --tp 8`;
        cmd += ` \\\n  --ep 8`;
      } else {
        cmd += ` \\\n  --tp 4`;
      }

      // Add tool call parser if enabled
      if (toolcall === 'enabled') {
        cmd += ` \\\n  --tool-call-parser minimax-m2`;
      }

      // Add thinking parser if enabled
      if (thinking === 'enabled') {
        cmd += ` \\\n  --reasoning-parser minimax-append-think`;
      }

      cmd += ` \\\n  --trust-remote-code`;
      cmd += ` \\\n  --mem-fraction-static 0.85`;

      // Add AMD-specific backend configurations
      if (isAMD) {
        cmd += ` \\\n  --attention-backend triton`;
      }

      return cmd;
    }
  };

  return <ConfigGenerator config={config} />;
};

export default MiniMaxM25ConfigGenerator;
