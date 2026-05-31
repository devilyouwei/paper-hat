import { createRouter, createWebHashHistory } from "vue-router";

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", redirect: "/chat" },
    {
      path: "/chat",
      name: "chat",
      component: () => import("@/views/ChatView.vue"),
      meta: { title: "Chat" },
    },
    {
      path: "/models",
      name: "models",
      component: () => import("@/views/ModelsView.vue"),
      meta: { title: "Models" },
    },
    {
      path: "/embedding-models",
      name: "embedding-models",
      component: () => import("@/views/EmbeddingModelsView.vue"),
      meta: { title: "Embeddings" },
    },
    {
      path: "/memory",
      name: "memory",
      component: () => import("@/views/MemoryView.vue"),
      meta: { title: "Memory" },
    },
  ],
});

export default router;
