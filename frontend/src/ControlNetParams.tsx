import { useSnapshot } from "valtio";
import { ControlNetConditionParams } from "./ControlNetConditionParams";
import { CollapsibleContainer } from "./components/Container";
import { Tab, TabList, TabNewButton, TabPanel, TabPanels, Tabs } from "./components/Tabs";
import { useControlNetModels } from "./queries";
import { ControlNetConditionParamsState, ControlNetParamsState } from "./schema";

interface ControlNetParamsProps {
  state: ControlNetParamsState;
  baseModelType: string | undefined;
}

export const ControlNetParams = ({ state, baseModelType }: ControlNetParamsProps) => {
  const snap = useSnapshot(state);
  const queryControlNetModels = useControlNetModels(baseModelType);

  function handleNewCondition(): void {
    state.conditions.push(new ControlNetConditionParamsState());
    state.activeTab = state.conditions.length - 1;
  }

  function handleRemoveCondition(index: number): void {
    state.conditions.splice(index, 1);
    if (state.activeTab >= index) {
      state.activeTab = Math.max(state.activeTab - 1, 0);
    }
  }

  if (queryControlNetModels.data?.length == 0) return <></>;

  return (
    <CollapsibleContainer
      label="ControlNet"
      hasSwitch={true}
      isOpen={snap.isOpen}
      isEnabled={snap.isEnabled}
      onIsOpenChange={(x) => (state.isOpen = x)}
      onIsEnabledChange={(x) => (state.isEnabled = x)}
    >
      <Tabs activeTab={snap.activeTab} onTabChange={(x) => (state.activeTab = x)}>
        <TabList>
          {snap.conditions.map((condition, index) => (
            <Tab key={condition.id} index={index} onClose={handleRemoveCondition}>
              {(index + 1).toString()}
            </Tab>
          ))}
          <TabNewButton onClick={handleNewCondition} />
        </TabList>

        <TabPanels>
          {snap.conditions.map((condition, index) => (
            <TabPanel key={condition.id} index={index}>
              <ControlNetConditionParams state={state.conditions[index]} baseModelType={baseModelType} />
            </TabPanel>
          ))}
        </TabPanels>
      </Tabs>
    </CollapsibleContainer>
  );
};
