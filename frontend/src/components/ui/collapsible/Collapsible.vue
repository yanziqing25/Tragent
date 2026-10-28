<script setup lang="ts">
import {
  CollapsibleRoot,
  type CollapsibleRootEmits,
  type CollapsibleRootProps,
  useForwardPropsEmits,
} from 'reka-ui'

const props = defineProps<CollapsibleRootProps>()
const emits = defineEmits<CollapsibleRootEmits>()

const forwarded = useForwardPropsEmits(props, emits)
</script>

<template>
  <CollapsibleRoot v-slot="{ open }" v-bind="forwarded">
    <slot :open="open" />
  </CollapsibleRoot>
</template>
